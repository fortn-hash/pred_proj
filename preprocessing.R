rm(list = ls())

library(readxl)
library(dplyr)
library(mice)

df <- read_excel( 'input.xlsx', sheet='hospital')
df = df %>%
  filter(DiagnosisYear >= 2016, DiagnosisYear <= 2025,
         Age >= 18, APL == 0) %>%
  mutate(IDH = case_when(IDH1 == 1 | IDH2 == 1 ~ 1,
                         IDH1 == 0 & IDH2 == 0 ~ 0,
                         TRUE ~ NA_real_)) %>%
  filter(NGS == 1, !is.na(IDH)) %>%
  mutate(Sex = factor(Sex, levels = c('F', 'M'), labels = c('0', '1')),
         cohort = if_else(DiagnosisYear <= 2021, 'training',
                          'temporal_validation'))
cnt = df %>% group_by(cohort) %>%
  summarise(n = n(), positive = sum(IDH == 1), negative = sum(IDH == 0),
            .groups = 'drop')
print(cnt)

vars = c('Age', 'Sex', 'NEUT', 'Hb', 'PLT', 'ALB', 'ALT', 'AST',
         'CREA', 'LDH', 'FIB', 'UREA', 'EGFR', 'PBBABS', 'BMB', 'BMI')
x <- as.data.frame(df[,vars])
train = df$cohort == "training"
info = df %>% select(SampleID, DiagnosisYear, IDH)
######
set.seed(100)
init = x
for (v in vars) {
  donor = x[train & !is.na(x[[v]]), v]
  init[is.na(x[[v]]), v] = sample(donor, sum(is.na(x[[v]])), replace = TRUE)
}

meth <- make.method(x)
pred=make.predictorMatrix(x)
imp = mice(x, m=10, maxit = 30, method = meth,
                 predictorMatrix=pred, ignore = !train, data.init = init,
           remove.constant = FALSE, remove.collinear = FALSE,
           seed = 100, printFlag = FALSE)

dir.create('processed', showWarnings = FALSE)
write.csv(cnt, 'processed/n.csv', row.names = FALSE)
dat = dat1 = vector('list', 10)
for (i in 1:10) {
    df1 <- complete(imp,i)
  df1$Sex = as.integer(as.character(df1$Sex))
  df1 = cbind(info, df1)
  dat[[i]] = df1[train, ]
  dat1[[i]] = df1[!train, ]
  write.csv(dat[[i]], paste0('processed/train', i, '.csv'),
            row.names = FALSE)
  write.csv(dat1[[i]],
            paste0('processed/validation', i, '.csv'), row.names = FALSE)
}
