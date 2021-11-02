library(pwr)


# we expect a large effect size as our previous paper showed medium to large effect (0.7 to 1.35) sizes on reaction times.
cohen_val = cohen.ES(test = "anov", size = "large")

pwr.anova.test(k = 3, f= cohen_val$effect.size, sig.level = 0.05, power = 0.80)

# the minimum effect size seen in the Impact of Task paper was 0.43 
pwr.anova.test(k = 3, f= 0.43, sig.level = 0.05, power = 0.80)
