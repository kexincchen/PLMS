rm(list = ls());
setwd("/Users/bsyiem/Documents/UnityProjects/PLMS/RScripts");

# for reading xlsx files
library("xlsx")

#used to easily plot qqplots and Anova
library("car");
library("ez");

#for two way Anova
library(tidyverse);
library(ggpubr);
library(rstatix);
library(scales)

#Multilevel Linear Model
library(lme4);
library(lmerTest);

#data formatting
library(dplyr);
library(tidyr);
library(data.table);

library(plyr);

library(MASS);

# for equivalence test
library("TOSTER");

library(pwr);

library(rje);
#####################################
# PREPARING DATA
#####################################
mpath <- "../PythonScripts/Data/Participant_Data/"
mdata_file <- "data.xlsx"

my_data <- read.xlsx(paste(mpath,mdata_file, sep = ""), sheetIndex = 1)

my_data$pid <- as.factor(my_data$pid)
my_data$gender <- as.factor(my_data$gender)
my_data$mid_air <- as.factor(my_data$mid_air)
my_data$problem_solving <- as.factor(my_data$problem_solving)

my_data$q1 <- as.factor(my_data$q1)
my_data$q2 <- as.factor(my_data$q2)
my_data$q3 <- as.factor(my_data$q3)
my_data$q4 <- as.factor(my_data$q4)
my_data$q5 <- as.factor(my_data$q5)


#RATE CORRECT SCORE -> "a comparison of methods to combine speed and accuracy measures of performance: a rejoining on the binning procedure" - Andre Vandierendonck
#my_data$rcs <- sqrt(my_data$rcs) # since we have 0 values, we cannot use log transform and use a sqrt transform instead

# maybe use boxcox transformation

#Maybe RCS from number of answes/second to number of answers/minute 
my_data$rcs <- my_data$rcs*60

#discard participants
to_discard <- c(5,7,25,29,41,47,62)
my_data <- my_data[!(my_data$pid %in% to_discard),]

#rcs_outlier_discard <- c(2,11,28,34,39,64)
#my_data <- my_data[!(my_data$pid %in% rcs_outlier_discard),]

#rcs_outlier_discard <- c(64)
#my_data <- my_data[!(my_data$pid %in% rcs_outlier_discard),]

summary(my_data)


completionTime_summary <- my_data %>% group_by(method) %>% get_summary_stats(completion_time, type = "mean_sd")
completionTime_summary

#Rate Correct Score (rcs) -> number of correct responses per second of activity
rcs_summary <- my_data %>% group_by(method) %>% get_summary_stats(rcs, type = "mean_sd")
rcs_summary

#number of correct responses grouped by method
aggregate(my_data$all_q, by = list(my_data$method), FUN = sum)

#####################################
#ASSUMPTIONS
#####################################

#outlier
my_data %>% group_by(method) %>% identify_outliers(completion_time);
my_data %>% group_by(method) %>% identify_outliers(rcs);

#shapiro test
my_data %>% group_by(method) %>% shapiro_test(completion_time);
my_data %>% group_by(method) %>% shapiro_test(rcs);

#test homogeneity
leveneTest(my_data$completion_time~my_data$method);
leveneTest(my_data$rcs~my_data$method);
  
#qqplots
ggqqplot(my_data,"completion_time", ggtheme = theme_bw()) + 
  facet_grid(~method, labeller = "label_both");

ggqqplot(my_data,"rcs", ggtheme = theme_bw()) + 
  facet_grid(~method, labeller = "label_both");

#####################################
#ANOVA
#####################################
ezANOVA(data = my_data, dv = .(completion_time), wid = .(pid), between = .(method), type = 3, detailed = T);
ezANOVA(data = my_data[my_data$all_q == 5,], dv = .(completion_time), wid = .(pid), between = .(method), type = 3, detailed = T);

ez.rcs <- ezANOVA(data = my_data, dv = .(rcs), wid = .(pid), between = .(method), type = 3, detailed = T);
ez.rcs

# Effect size
ez.rcs$ANOVA$SSn/(ez.rcs$ANOVA$SSn + ez.rcs$ANOVA$SSd)


#####################################
#Kruskal-Wallis
#####################################

kruskal.test(rcs~method, data = my_data)

# Effect size kruskal wallis test eta^2
my_data %>% kruskal_effsize(rcs~method)

#####################################
#Plots
#####################################
my_data_plot <- my_data

# renaming for plot
revalue(my_data_plot$method, c("PHYSICAL" = "Paper")) -> my_data_plot$method;
revalue(my_data_plot$method, c("AR" = "Unassisted AR")) -> my_data_plot$method;
revalue(my_data_plot$method, c("RL_CLUSTER" = "Assisted AR")) -> my_data_plot$method;


means <- aggregate(rcs ~ method, my_data_plot, mean)
sd <- aggregate(rcs ~ method, my_data_plot, sd)

means_sd <- means
means_sd$sd <- sd$rcs

colnames(means_sd) <- c("Condition", "rcs", "sd")

p <- ggplot(means_sd, aes(x = Condition, y = rcs, fill = Condition)) +
     geom_bar(stat = "identity", colour = "black", width = 0.8) +
     geom_errorbar(aes(ymin = rcs-sd, ymax = rcs + sd), width = 0.2, position = position_dodge(0.9)) +
     geom_text(aes(label = round(rcs, digits = 3)), y = 0.006) +
     #geom_text(aes(label = comma(rcs)), y = 0.006) + # for scientific labels
     ylab(label = "Rate Correct Score") +
     theme_minimal() +
     scale_fill_brewer(palette = "Accent") +
     scale_x_discrete(limits = c("Paper", "Unassisted AR", "Assisted AR")) +
     scale_y_continuous(labels = comma) # for scientific labels

p

p <- ggplot(means_sd, aes(x = Condition, y = rcs)) +
  geom_bar(stat = "identity", colour = "black", fill = "#CCCCCC", width = 0.8) +
  geom_errorbar(aes(ymin = rcs-sd, ymax = rcs + sd), width = 0.2, position = position_dodge(0.9)) +
  geom_text(aes(label = round(rcs, digits = 3)), y = 0.006) +
  #geom_text(aes(label = comma(rcs)), y = 0.006) + # for scientific labels
  ylab(label = "Rate Correct Score") +
  theme_minimal() +
  scale_fill_brewer(palette = "Accent") +
  scale_x_discrete(limits = c("Paper", "Unassisted AR", "Assisted AR")) +
  scale_y_continuous(labels = comma) # for scientific labels

p
#####################################
# Linear models
#####################################
# first group by method
#my_data$AR <- (my_data$method == 'AR') + 0
#my_data$PHYSICAL <- (my_data$method == 'PHYSICAL') + 0
#my_data$RL_CLUSTER <- (my_data$method == 'RL_CLUSTER') + 0

ggplot(data = my_data,
       aes(x = method,
           y = completion_time)) +
  geom_point(size = 1.2,
             alpha = 0.8) +
  theme_minimal()

model <- lmer(completion_time ~ 1 + method + age + gender + education + mid_air + problem_solving + (1 | pid), data = my_data)
model <- lmer(completion_time ~ 1 + method + (1 | pid), data = my_data)
summary(model)

#####################################
# ANCOVA
# https://www.datanovia.com/en/lessons/ancova-in-r/#:~:text=The%20Analysis%20of%20Covariance%20(ANCOVA,two%20or%20more%20independent%20groups.
#####################################

#dependent <- completion_time
#group or IVs <- method
#covariates <- age, gender, education, mid_air, problem_solving

##########
#Interactions I care about
##########
# all covariate interaction with IVs


#linearity assumption

#ggscatter(
#  my_data, x = "age", y = "completion_time",
#  color = "method", add = "reg.line"
#)+
#  stat_regline_equation(
#    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~"), color = method)
#  )

#age
ggscatter(
  my_data, x = "age", y = "completion_time",
  facet.by  = "method", 
  short.panel.labs = FALSE
)+
  stat_smooth(method = "loess", span = 0.9)

#gender
ggscatter(
  my_data, x = "gender", y = "completion_time",
  facet.by  = "method", 
  short.panel.labs = FALSE
)+
  stat_smooth(method = "loess", span = 0.9)

#education
ggscatter(
  my_data, x = "education", y = "completion_time",
  facet.by  = "method", 
  short.panel.labs = FALSE
)+
  stat_smooth(method = "loess", span = 0.9)

#mid_air
ggscatter(
  my_data, x = "mid_air", y = "completion_time",
  facet.by  = "method", 
  short.panel.labs = FALSE
)+
  stat_smooth(method = "loess", span = 0.9)

#problem_solving
ggscatter(
  my_data, x = "problem_solving", y = "completion_time",
  facet.by  = "method", 
  short.panel.labs = FALSE
)+
  stat_smooth(method = "loess", span = 0.9)

#Homogeneity of regression slopes
my_data %>%
  anova_test(completion_time ~ 
               age + 
               gender + 
               education + 
               mid_air + 
               problem_solving + 
               method +
               age * method +
               gender * method +
               education * method +
               mid_air * method + 
               problem_solving * method +
               age * gender * method +
               age * education * method +
               age * mid_air * method +
               age * problem_solving * method +
               gender * education * method +
               gender * mid_air * method + 
               gender * problem_solving * method +
               education * mid_air * method +
               education * problem_solving * method +
               mid_air * problem_solving * method +
               age * gender * education * method +
               age * gender * mid_air * method +
               age * gender * problem_solving * method +
               age * education * mid_air * method +
               age * education * problem_solving * method +
               age * mid_air * problem_solving * method +
               gender * education * mid_air * method + 
               gender * education * problem_solving * method +
               gender * mid_air * problem_solving * method +
               education * mid_air * problem_solving * method + 
               age * gender * education * mid_air * problem_solving * method)

my_data %>%
  anova_test(rcs ~ 
               age + 
               gender + 
               education + 
               mid_air + 
               problem_solving + 
               method +
               age * method +
               gender * method +
               education * method +
               mid_air * method + 
               problem_solving * method +
               age * gender * method +
               age * education * method +
               age * mid_air * method +
               age * problem_solving * method +
               gender * education * method +
               gender * mid_air * method + 
               gender * problem_solving * method +
               education * mid_air * method +
               education * problem_solving * method +
               mid_air * problem_solving * method +
               age * gender * education * method +
               age * gender * mid_air * method +
               age * gender * problem_solving * method +
               age * education * mid_air * method +
               age * education * problem_solving * method +
               age * mid_air * problem_solving * method +
               gender * education * mid_air * method + 
               gender * education * problem_solving * method +
               gender * mid_air * problem_solving * method +
               education * mid_air * problem_solving * method + 
               age * gender * education * mid_air * problem_solving * method)
"
my_data %>%
  anova_test(completion_time ~ method*age*gender*education*mid_air*problem_solving)

my_data %>%
  anova_test(completion_time ~ (age + gender + education + mid_air + problem_solving)*method)

my_data %>%
  anova_test(completion_time ~ age + gender + education + mid_air + problem_solving + method + age:method + education:method + mid_air:method + problem_solving:method)

my_data %>%
  anova_test(completion_time ~ method*age)

my_data %>%
  anova_test(completion_time ~ method*gender)

my_data %>%
  anova_test(completion_time ~ method*education)

my_data %>%
  anova_test(completion_time ~ method*mid_air)

my_data %>%
  anova_test(completion_time ~ method*problem_solving)
"

#Normality
# Fit the model, the covariate goes first
model <- lm(completion_time ~ age + gender + education + mid_air + problem_solving + method, data = my_data)

model <- lm(rcs ~ age + gender + education + mid_air + problem_solving + method, data = my_data)

#with interaction maybe
#model <- lm(completion_time ~ age + gender + education + mid_air + problem_solving + age:method + education:method + mid_air:method + problem_solving:method + method, data = my_data)
# Inspect the model diagnostic metrics
model.metrics <- augment(model) %>%
  select(-.hat, -.sigma, -.fitted, -.se.fit) # Remove details
head(model.metrics, 3)

# Assess normality of residuals using shapiro wilk test
shapiro_test(model.metrics$.resid)

#Homogeneity of variances
levene_test(.resid ~ method, data = model.metrics)

#Outliers
model.metrics %>% 
  filter(abs(.std.resid) > 3) %>%
  as.data.frame()

#Computation
res.aov <- my_data %>% 
  anova_test(completion_time ~ age + gender + education + mid_air + problem_solving + method, type = 2)
get_anova_table(res.aov)

res.aov <- my_data %>% 
  anova_test(rcs ~ age + gender + education + mid_air + problem_solving + method, type = 2)
get_anova_table(res.aov)

#with interactions
# education:method is removed due to linear dependence
#res.aov <- my_data %>% 
#  anova_test(completion_time ~ age + gender + education + mid_air + problem_solving + age:method + mid_air:method + problem_solving:method + method, type = 3)
#get_anova_table(res.aov)

##########################################
# NASA TLX
##########################################

#ANOVA
my_data %>% group_by(method) %>% shapiro_test(avg_tlx);
leveneTest(my_data$avg_tlx~my_data$method);
ggqqplot(my_data,"avg_tlx", ggtheme = theme_bw()) + 
  facet_grid(~method, labeller = "label_both");

ez.tlx <- ezANOVA(data = my_data, dv = .(avg_tlx), wid = .(pid), between = .(method), type = 3, detailed = T);
ez.tlx$ANOVA$SSn/(ez.tlx$ANOVA$SSn + ez.tlx$ANOVA$SSd)

lm.model.tlx <- lmList(rcs ~ avg_tlx | method, data=my_data)
lm.model.tlx

lm.model.tlx.ph <- lm(rcs~avg_tlx, data = my_data[my_data$method == "PHYSICAL",])
summary(lm.model.tlx.ph)

lm.model.tlx.ar <- lm(rcs~avg_tlx, data = my_data[my_data$method == "AR",])
summary(lm.model.tlx.ar)

lm.model.tlx.rl <- lm(rcs~avg_tlx, data = my_data[my_data$method == "RL_CLUSTER",])
summary(lm.model.tlx.rl)

##########################################
# the ones that have a significant effect from the NASA TLX  --- using YATINI
#PHYSICAL DEMAND
my_data %>% group_by(method) %>% shapiro_test(physical_demand);
leveneTest(my_data$physical_demand~my_data$method);
kruskal.test(physical_demand~method, data = my_data)

my_data %>% kruskal_effsize(physical_demand~method)

#Mann-Whitney Test
my_data  %>% wilcox_test(physical_demand~method, paired = F)
p<- ggboxplot(
  my_data, x = "method", y = "physical_demand"
)
p

wilcox.test(my_data[my_data$method == "PHYSICAL",]$physical_demand, my_data[my_data$method == "AR",]$physical_demand)
wilcox.test(my_data[my_data$method == "PHYSICAL",]$physical_demand, my_data[my_data$method == "RL_CLUSTER",]$physical_demand)
wilcox.test(my_data[my_data$method == "AR",]$physical_demand, my_data[my_data$method == "RL_CLUSTER",]$physical_demand)

# FOR EXACT VALUES
library(coin)
group_paper <- my_data[my_data$method == "PHYSICAL",]$physical_demand
group_ar <- my_data[my_data$method == "AR",]$physical_demand
group_rl <- my_data[my_data$method == "RL_CLUSTER",]$physical_demand

g <- factor(c(rep("PHYSICAL",length(group_paper)), rep("AR",length(group_ar))))
v <- c(group_paper,group_ar)
wilcox_test(v~g, distribution = "exact")

g <- factor(c(rep("PHYSICAL",length(group_paper)), rep("RL_CLUSTER",length(group_rl))))
v <- c(group_paper,group_rl)
wilcox_test(v~g, distribution = "exact")

g <- factor(c(rep("AR",length(group_ar)), rep("RL_CLUSTER",length(group_rl))))
v <- c(group_ar,group_rl)
wilcox_test(v~g, distribution = "exact")

#effect size for significant
paper_ar <- 2.5031/sqrt(length(group_paper) + length(group_ar))
paper_ar

paper_rl <- 2.7691/sqrt(length(group_paper) + length(group_rl))
paper_rl
##########################################
#EFFORT
my_data %>% group_by(method) %>% shapiro_test(effort);
leveneTest(my_data$effort~my_data$method);
res<-ezANOVA(data = my_data, dv = .(effort), wid = .(pid), between = .(method), type = 3, detailed = T);
res
#effect_size ANOVA
res$ANOVA$SSn/(res$ANOVA$SSn+res$ANOVA$SSd)

#Unpaired T-Test
my_data %>% t_test(effort ~ method, paired = F) # I dont think I am running this command correctly

p<- ggboxplot(
  my_data, x = "method", y = "effort"
)
p

t.test(my_data[my_data$method == "PHYSICAL",]$effort, my_data[my_data$method == "AR",]$effort, var.equal = F)
t.test(my_data[my_data$method == "PHYSICAL",]$effort, my_data[my_data$method == "RL_CLUSTER",]$effort, var.equal = F)
t.test(my_data[my_data$method == "AR",]$effort, my_data[my_data$method == "RL_CLUSTER",]$effort, var.equal = F)

#effect size for significants
library(MBESS)
abs(smd(my_data[my_data$method == "PHYSICAL",]$effort, my_data[my_data$method == "AR",]$effort))


########################
# IQR
#########################

iqr_rcs_agent <- IQR(my_data[my_data$method == "RL_CLUSTER",]$rcs)

quantiles <- quantile(my_data[my_data$method == "RL_CLUSTER",]$rcs)


first_quartile <- my_data[my_data$method == "RL_CLUSTER" & my_data$rcs <= quantiles[[2]],]
last_quartile <- my_data[my_data$method == "RL_CLUSTER" & my_data$rcs >= quantiles[[4]],]

###################################
# diff in performance between top 25% of each group
###################################

quantiles_p <- quantile(my_data[my_data$method == "PHYSICAL",]$rcs)
quantiles_a <- quantile(my_data[my_data$method == "AR",]$rcs)

last_quartile_p <- my_data[my_data$method == "PHYSICAL" & my_data$rcs >= quantiles_p[[4]],]
last_quartile_a <- my_data[my_data$method == "AR" & my_data$rcs >= quantiles_a[[4]],]

all_top = rbind(last_quartile,last_quartile_p,last_quartile_a)

summary(all_top)


completionTime_summary <- all_top %>% group_by(method) %>% get_summary_stats(completion_time, type = "mean_sd")
completionTime_summary

#Rate Correct Score (rcs) -> number of correct responses per second of activity
rcs_summary <- all_top %>% group_by(method) %>% get_summary_stats(rcs, type = "mean_sd")
rcs_summary

ez.rcs.top <- ezANOVA(data = all_top, dv = .(rcs), wid = .(pid), between = .(method), type = 3, detailed = T);
ez.rcs.top

t.test(all_top[all_top$method == "PHYSICAL",]$rcs, all_top[all_top$method == "AR",]$rcs, var.equal = F)
t.test(all_top[all_top$method == "PHYSICAL",]$rcs, all_top[all_top$method == "RL_CLUSTER",]$rcs, var.equal = F)
t.test(all_top[all_top$method == "RL_CLUSTER",]$rcs, all_top[all_top$method == "AR",]$rcs, var.equal = F)

#########################
# Performance of participants with and without reports of interaction issues
#########################

AR_issue = c(3, 8, 17, 27, 32, 35, 48, 55, 64)
RL_issue = c(1, 7, 11, 16, 22, 34, 44, 46, 57, 61, 66)

AR_issue_data = my_data[my_data$pid %in% AR_issue,]
RL_issue_data = my_data[my_data$pid %in% RL_issue,]

AR_non_issue_data = my_data[!(my_data$pid %in% AR_issue) & my_data$method == "AR",]
RL_non_issue_data = my_data[!(my_data$pid %in% RL_issue) & my_data$method == "RL_CLUSTER",]

issue_data = rbind(AR_issue_data, RL_issue_data)
non_issue_data = rbind(AR_non_issue_data, RL_non_issue_data)

issue_data$interaction_issue = rep(TRUE, nrow(issue_data))
non_issue_data$interaction_issue = rep(FALSE, nrow(non_issue_data))

AR_data = rbind(issue_data[issue_data$method == "AR",], non_issue_data[non_issue_data$method == "AR",])
RL_data = rbind(issue_data[issue_data$method == "RL_CLUSTER",], non_issue_data[non_issue_data$method == "RL_CLUSTER",])

#########################
# FOR Unassisted AR condition
#########################
AR_data %>% group_by(interaction_issue) %>% identify_outliers(rcs);
AR_data %>% group_by(interaction_issue) %>% shapiro_test(rcs);
leveneTest(AR_data$rcs~AR_data$interaction_issue);

#kruskal.test(rcs~interaction_issue, data = AR_data)
ezANOVA(data = AR_data, dv = .(rcs), wid = .(pid), between = .(interaction_issue), type = 3, detailed = T);

#########################
# FOR Assisted AR condition
#########################
RL_data %>% group_by(interaction_issue) %>% identify_outliers(rcs);
RL_data %>% group_by(interaction_issue) %>% shapiro_test(rcs);
leveneTest(RL_data$rcs~RL_data$interaction_issue);

kruskal.test(rcs~interaction_issue, data = RL_data)
#ezANOVA(data = RL_data, dv = .(rcs), wid = .(pid), between = .(interaction_issue), type = 3, detailed = T);

#########################
# FOR ALL AR CONDITIONS
#########################
new_data = rbind(issue_data, non_issue_data)
new_data %>% group_by(interaction_issue) %>% identify_outliers(rcs);
new_data %>% group_by(interaction_issue) %>% shapiro_test(rcs);
leveneTest(new_data$rcs~new_data$interaction_issue);

kruskal.test(rcs~interaction_issue, data = new_data)


#############################################################################################################################
#########################
# AGENT vs RANDOM SOLVER -  NORMALIZED (Including training rewards)
#########################

agent_rewards <- c(0.7455903704038609, 0.7381049268285339, 0.8132607898398908, 0.8007214159043061, 0.8305828096034729, 0.725334458661762, 0.7870868562644114, 0.8022084570441045, 0.71024300060788, 0.7308405299091203, 0.7295946305757746, 0.7282231365515722, 0.9609852652308679, 0.7435507126242767, 0.8197866899770397, 0.8260865196706392, 0.7682175099094196, 0.7197530305998914, 0.7318302160731046, 0.7285948968365206)
random_solver_rewards <- c(0.03659326912932653, 0.08135019316463397, 0.03820088117235062, 0.11358783841489345, 0.057572606290786525, 0.08399270546035469, 0.16688520142876404, 0.05021778119395276, 0.11120656307616486, 0.07671323717803766, 0.11037763811648121, 0.09349771166473163, 0.116808086288576, 0.17807820027831728, 0.0, 0.06820294092528079, 0.12012378612731177, 0.044495687078315325, 0.14885482760872726, 0.04233545839550147)

agent_rewards <- data.frame(rep("AG",length(agent_rewards)),agent_rewards)
random_solver_rewards <- data.frame(rep("RS",length(random_solver_rewards)),random_solver_rewards)

colnames(agent_rewards) <- c('Solver','Reward')
colnames(random_solver_rewards) <- c('Solver','Reward')

all_rewards <- rbind(agent_rewards,random_solver_rewards)

reward_summary <- all_rewards %>% group_by(Solver) %>% get_summary_stats(Reward, type = "mean_sd")
reward_summary

# ASSUMPTIONS
#outlier
all_rewards %>% group_by(Solver) %>% identify_outliers(Reward);

#shapiro test
all_rewards %>% group_by(Solver) %>% shapiro_test(Reward);

#test homogeneity
leveneTest(all_rewards$Reward~all_rewards$Solver);

#qqplots
ggqqplot(all_rewards,"Reward", ggtheme = theme_bw()) + 
  facet_grid(~Solver, labeller = "label_both");

#kruskal.test(Reward~Solver, data = all_rewards)
wilcox.test(all_rewards[all_rewards$Solver == "AG",]$Reward, all_rewards[all_rewards$Solver == "RS",]$Reward)
wilcox_test(all_rewards$Reward~factor(all_rewards$Solver), distribution = "exact")

effect_size = 5.41/sqrt(nrow(all_rewards))
effect_size

#########################
# AGENT VS RANDOM SOLVER (NON-NORMALIZED) -- FIRST TEST - data does not include normalization with training rewards
#########################

agent_rewards <- c(-590.500, -1530.950, -1703.200, -820.000, -1779.250, -736.750, -910.150, -1686.100, -687.200, -1242.050, -675.600, -832.350, -1461.200, -1567.900, -586.650, -1707.500, -621.550, -1643.250, -1469.100, -1682.850)
random_solver_rewards <- c(-8019.950000000002,-8221.850000000002,-8374.450000000003,-8248.2,-7982.500000000002,-9035.050000000003,-8291.9,-6913.450000000004,-7800.199999999999,-9149.349999999993,-9235.399999999994,-8317.850000000002,-8081.399999999993,-8528.149999999998,-7034.200000000003,-7933.9000000000015,-8706.80000000001,-8542.199999999997,-8211.850000000004,-7448.300000000006)

agent_rewards <- data.frame(rep("AG",length(agent_rewards)),agent_rewards)
random_solver_rewards <- data.frame(rep("RS",length(random_solver_rewards)),random_solver_rewards)

colnames(agent_rewards) <- c('Solver','Reward')
colnames(random_solver_rewards) <- c('Solver','Reward')

all_rewards <- rbind(agent_rewards,random_solver_rewards)

reward_summary <- all_rewards %>% group_by(Solver) %>% get_summary_stats(Reward, type = "mean_sd")
reward_summary

# ASSUMPTIONS
#outlier
all_rewards %>% group_by(Solver) %>% identify_outliers(Reward);

#shapiro test
all_rewards %>% group_by(Solver) %>% shapiro_test(Reward);

#test homogeneity
leveneTest(all_rewards$Reward~all_rewards$Solver);

#qqplots
ggqqplot(all_rewards,"Reward", ggtheme = theme_bw()) + 
  facet_grid(~Solver, labeller = "label_both");

#kruskal.test(Reward~Solver, data = all_rewards)
wilcox.test(all_rewards[all_rewards$Solver == "AG",]$Reward, all_rewards[all_rewards$Solver == "RS",]$Reward)
wilcox_test(all_rewards$Reward~factor(all_rewards$Solver), distribution = "exact")

effect_size = 5.41/sqrt(nrow(all_rewards))
effect_size