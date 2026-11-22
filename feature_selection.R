rm(list = ls())
library(glmnet)
library(e1071)
library(randomForest)
library(pROC)

vars = c('Age', 'Sex', 'NEUT', 'Hb', 'PLT', 'ALB', 'ALT', 'AST',
         'CREA', 'LDH', 'FIB', 'UREA', 'EGFR', 'PBBABS', 'BMB', 'BMI')
methods = c('LR-RFE', 'SVM-RFE', 'MI', 'RF', 'mRMR')
n_keep = ceiling(length(vars) / 2)
freq <- 5:10
min_vote=3

rfe=function(x, y, k, model) {
  keep <- colnames(x)
  while (length(keep) > k) {
    z =x[, keep, drop = FALSE]
    if (model =='LR') {
      fit = glmnet(z, y, family = 'binomial', alpha = 0,
                   lambda = 1 / nrow(z), standardize = FALSE)
      b = as.numeric(coef(fit))[-1]
    } else {
      fit = svm(z, factor(y), kernel = 'linear', scale = FALSE)
      b = drop(t(fit$coefs) %*% fit$SV)
    }
    keep = keep[-which.min(b^2)]
  }
  keep
}
# mutual information
mi_discrete = function(dat, b) {
  p =prop.table(table(dat, b))
    p0=outer(rowSums(p),colSums(p))
  ok = p >0
  sum(p[ok] * log(p[ok] / p0[ok]))
}

mi_continuous= function(x, y, k = 3) {
  if (length(unique(x)) < 2) return(0)
  x = x/ sd(x)
  x = x +rnorm(length(x), sd = 1e-10 * max(1, mean(abs(x))))
  nc = as.numeric(table(y)[as.character(y)])
  keep = nc > 1
  x = x[keep]; y = y[keep]; nc = nc[keep]
  n = length(x)
  dist <- abs(outer(x,x,"-"))
  rad = nn = numeric(n)
  for (grp in unique(y)) {
    j = which(y == grp)
    nk = min(k, length(j) - 1)
    dist1 = dist[j, j, drop = FALSE]
    diag(dist1) = Inf
    rad[j] = apply(dist1, 1, function(v) sort(v)[nk])
    nn[j] = nk
  }
  cnt = rowSums(dist < rad)
  max(0, digamma(n) + mean(digamma(nn)) -
        mean(digamma(nc)) - mean(digamma(cnt)))
}

# mRMR
mrmr = function(x, y, k) {
  z = x
  for (v in colnames(x)) {
    if (v == 'Sex') next
    brk = unique(quantile(x[, v], seq(0, 1, length.out = 21)))
    z[, v] = if (length(brk) <2) 0 else
      cut(x[, v], breaks = brk, include.lowest = TRUE, labels = FALSE)
  }
  rel =apply(z, 2, mi_discrete, b = y)
  red = matrix(0, ncol(z), ncol(z))
  for (j in 1:ncol(z)) {
    for (l in 1:ncol(z)) red[j, l] = mi_discrete(z[, j], z[, l])
  }
  idx = which.max(rel)
  while (length(idx)<k) {
    rest = setdiff(seq_len(ncol(z)), idx)
    score = rel[rest] - rowMeans(red[rest, idx, drop = FALSE])
      idx = c(idx, rest[which.max(score)])
  }
  colnames(x)[idx]
}

# read the 10 sets
dat <- lapply(1:10,function(i) {
  read.csv(paste0('processed/train', i, '.csv'))
})
sel = array(FALSE, c(length(vars), length(methods), 10),
                 dimnames = list(vars, methods, paste0('imp', 1:10)))


for (i in 1:10) {
  df = dat[[i]]
  x = as.matrix(df[, vars])
  y = df$IDH
  s = apply(x, 2, sd)
  s[s == 0] = 1
  xs = scale(x, center = TRUE, scale = s)

  set.seed(100 + i)
  sel[, 'LR-RFE', i] = vars %in% rfe(xs, y, n_keep, 'LR')
  sel[, 'SVM-RFE', i] = vars %in% rfe(xs, y, n_keep, 'SVM')
  mi = sapply(vars, function(v) {
    if (v == 'Sex') mi_discrete(x[, v], y) else mi_continuous(x[, v], y)
  })
  idx = head(order(mi, decreasing = TRUE), n_keep)
  sel[idx, 'MI', i] = TRUE
  fit = randomForest(x, factor(y))
  rf_imp = drop(importance(fit, type = 2))
  idx = head(order(rf_imp, decreasing = TRUE), n_keep)
  sel[idx, 'RF', i] = TRUE
  sel[, 'mRMR', i] = vars %in% mrmr(x, y, n_keep)
}
# count 
counts <- apply(sel,c(1, 2),sum)
dir.create('thresholds', showWarnings = FALSE)
write.csv(data.frame(variable = vars, counts, check.names = FALSE),
          'thresholds/inclusion.csv', row.names = FALSE)
y = dat[[1]]$IDH
fold = integer(length(y))
for (g in unique(y)) {
  j = which(y == g)
  fold[j] = sample(rep(1:5, length.out = length(j)))
}
write.csv(data.frame(SampleID = dat[[1]]$SampleID, fold = fold),
          'thresholds/folds.csv', row.names = FALSE)

# compare 
res = vote_res = data.frame()
oof = data.frame(SampleID = dat[[1]]$SampleID, IDH = y)
for (f in freq) {
  vote= counts >= f
  vars1 = vars[rowSums(vote) >= min_vote]
  v = data.frame(frequency = f, variable = vars, vote * 1, check.names = FALSE)
  v$votes = rowSums(vote)
  v$selected = v$votes >= min_vote
  vote_res = rbind(vote_res, v)

  auc1 = NA_real_
  if (length(vars1) > 0) {
      prob <- matrix(NA_real_,length(y),10)
    for (i in 1:10) {
      x = as.matrix(dat[[i]][, vars1, drop = FALSE])
      if (ncol(x) == 1) x = cbind(x, unused = 0)
      for (k in 1:5) {
        train = fold != k
        fit = glmnet(x[train, , drop = FALSE], y[train], family = 'binomial',
                alpha=0, lambda = 1 / sum(train), standardize=TRUE)
        prob[!train, i] = as.numeric(predict(fit,
          newx = x[!train, , drop = FALSE], type = 'response'))
      }
    }
    p = rowMeans(prob)
    auc1 = as.numeric(auc(roc(y, p, levels = c(0, 1), direction = '<', quiet = TRUE)))
    oof[[paste0('frequency_', f)]] = p
  } else {
    oof[[paste0('frequency_', f)]] = NA_real_
  }

  res = rbind(res, data.frame(
    frequency = paste0(f, '/10'), n_variables = length(vars1),
    variables = paste(vars1, collapse = ', '), cv_auc = auc1))
}

write.csv(res, 'thresholds/comparison.csv', row.names = FALSE)
write.csv(vote_res, 'thresholds/votes.csv', row.names = FALSE)
write.csv(oof, "thresholds/oof.csv", row.names=FALSE)
print(res)
