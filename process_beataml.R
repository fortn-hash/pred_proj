rm(list = ls())
library(readxl)
library(dplyr)
library(stringr)
library(mice)

# dir.create('beat')
# download.file('https://media.githubusercontent.com/media/biodev/beataml2.0_data/main/beataml_wv1to4_clinical.xlsx',
#               'beat/beataml_wv1to4_clinical.xlsx', mode = 'wb')
# download.file('https://media.githubusercontent.com/media/biodev/beataml2.0_data/main/beataml_wes_wv1to4_mutations_dbgap.txt',
#               'beat/beataml_wes_wv1to4_mutations_dbgap.txt', mode = 'wb')
df <- read_excel('beat/beataml_wv1to4_clinical.xlsx', sheet="summary")
mut <- read.delim('beat/beataml_wes_wv1to4_mutations_dbgap.txt', quote = '', comment.char = '',  check.names = FALSE, stringsAsFactors = FALSE)
f1 = function(x) {
    s <- tolower(trimws(as.character(x)))
  s[s %in% c('rare', 'occasional')] = '1'
  s[s == 'predominant'] = '80'
  as.numeric(str_extract(s, '[0-9]+\\.?[0-9]*'))
}

mut = mut %>% filter((symbol == 'IDH1' & str_detect(hgvsp_short, '^p\\.R132[A-Z]')) |
                 (symbol == 'IDH2' & str_detect(hgvsp_short, '^p\\.R(140|172)[A-Z]')))
df = df %>% mutate(
  PatientID = as.character(dbgap_subject_id), SampleID = dbgap_dnaseq_sample,
  wes = !is.na(SampleID) & manuscript_dnaseq == 'yes',
  IDH1 = if_else(wes, as.integer(SampleID %in% mut$dbgap_sample_id[mut$symbol == 'IDH1']), NA_integer_),
  IDH2 = if_else(wes, as.integer(SampleID %in% mut$dbgap_sample_id[mut$symbol == 'IDH2']), NA_integer_),
  IDH = case_when(IDH1 == 1 | IDH2 == 1 ~ 1L,
                  IDH1 == 0 & IDH2 == 0 ~ 0L, TRUE ~ NA_integer_),
  Age = coalesce(f1(ageAtDiagnosis), f1(ageAtSpecimenAcquisition)),
  Sex = case_when(consensus_sex == 'Female' ~ 0, consensus_sex == 'Male' ~ 1),
  NEUT = f1(wbcCount) * f1(`%.Neutrophils.in.PB`) / 100,
  Hb = f1(hemoglobin) * 10, PLT = f1(plateletCount),
  ALB = f1(albumin) * 10, ALT = f1(ALT), AST = f1(AST),
  CREA = f1(creatinine) * 88.4, LDH = f1(LDH),
  PBBABS = f1(wbcCount) * pmin(100, pmax(0, f1(`%.Blasts.in.PB`))) / 100,
  BMB = f1(`%.Blasts.in.BM`))

df = df %>%
  filter(str_starts(dxAtSpecimenAcquisition, 'ACUTE MYELOID'),
         diseaseStageAtSpecimenCollection == 'Initial Diagnosis',
         Age >= 18,
         !str_detect(coalesce(specificDxAtAcquisition, ''), regex('promyelocytic', TRUE)),
         !str_detect(coalesce(consensusAMLFusions, ''), 'PML-RARA'),
         wes, !is.na(IDH))
vars = c('Age', 'Sex', 'NEUT', 'Hb', 'PLT', 'ALB', 'ALT', 'AST',
       'CREA', 'LDH', 'PBBABS', 'BMB')
core = c('Age', 'Sex', 'LDH', 'PBBABS', 'BMB')
df = df %>% select(PatientID, SampleID, IDH1, IDH2, IDH, all_of(vars))
df$miss = rowSums(is.na(df[, c('LDH', 'PBBABS', 'BMB')]))
keep = complete.cases(df[, core])

# missing data
cnt <- data.frame()
for (k in 0:3) {
  for (s in c('exact', 'cumulative')) {
      idx = if(s == 'exact') df$miss == k else df$miss <= k
    cnt = bind_rows(cnt, data.frame(group = s, missing = k, n = sum(idx),
                                 positive = sum(df$IDH[idx] == 1), negative = sum(df$IDH[idx] == 0)))
  }
}
miss = data.frame(variable = vars, observed = colSums(!is.na(df[, vars])),
                missing = colSums(is.na(df[, vars])), row.names = NULL)

dir.create('ext',showWarnings=FALSE)
write.csv(cnt, 'ext/n.csv', row.names = FALSE)
write.csv(miss, 'ext/miss.csv', row.names = FALSE)
write.csv(df, 'ext/a.csv', row.names = FALSE)
write.csv(df[keep, ], 'ext/cc.csv', row.names = FALSE)
write.csv(df[!keep, ], 'ext/out.csv', row.names = FALSE)
print(cnt)

# MICE 
x = as.data.frame(df[, vars])
x$Sex = factor(x$Sex, levels = c(0, 1))
set.seed(100)
init = x
for (v in vars) {
  donor = x[!is.na(x[[v]]), v]
  init[is.na(x[[v]]), v] = sample(donor, sum(is.na(x[[v]])), replace = TRUE)
}
meth <- make.method(x)
pred = make.predictorMatrix(x)
imp <- mice(x, m=10, maxit=30, method = meth,
                  predictorMatrix=pred, data.init = init,
           remove.constant = FALSE, remove.collinear = FALSE,
           seed = 100, printFlag = FALSE)
info = df %>% select(PatientID, SampleID, IDH1, IDH2, IDH, miss)
for (i in 1:10) {
    dat <- complete(imp,i)
  dat$Sex = as.integer(as.character(dat$Sex))
  dat = cbind(info, dat)
  write.csv(dat,paste0('ext/x',i,'.csv'),row.names=FALSE)
}
