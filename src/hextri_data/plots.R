library(hextri)
library(hexbin)

covariate <- "Temperature"
dir.create(paste0(covariate, "/"))

# Beta Estimate Plot
png(filename=paste0(covariate, "/Beta_Estimates.png"), units="in", width=6, height=6, res=300)

estimates <- read.csv("estimates.csv")
estimates$class <- as.factor(estimates$class)

hextri(x=estimates$x, y=estimates$y, 
       class=estimates$class, colour=c("orange", "lightblue", "blue"),
       style="size", nbins=25, 
       xlab="Estimates with True Outcomes", 
       ylab="Estimates with Predicted Outcomes")

abline(0,1)

legend("topleft", col=c("orange", "light blue", "blue"), pch=15, 
       legend=c("No Correction", "Non-Parametric Bootstrap", "Parametric Bootstrap"))

title(main = "Beta Estimates")

dev.off()

# Standard Error Plot
png(filename=paste0(covariate, "/Standard_Errors.png"), units="in", width=6, height=6, res=300)

se <- read.csv("ses.csv")
se$class <- as.factor(se$class)

hextri(x=se$x, y=se$y, 
       class=se$class, colour=c("orange", "lightblue", "blue"),
       style="size", nbins=25, 
       xlab="Standard Error with True Outcomes", 
       ylab="Standard Error with Predicted Outcomes")

abline(0,1)

legend("topleft", col=c("orange", "light blue", "blue"), pch=15, 
       legend=c("No Correction", "Non-Parametric Bootstrap", "Parametric Bootstrap"))

title(main = "Standard Errors")

dev.off()


# T-Statistic Plot
png(filename=paste0(covariate, "/T_Statistics.png"), units="in", width=6, height=6, res=300)

tstat <- read.csv("tstats.csv")
tstat$class <- as.factor(tstat$class)

hextri(x=tstat$x, y=tstat$y, 
       class=tstat$class, colour=c("orange", "lightblue", "blue"),
       style="size", nbins=25, 
       xlab="T-statistic with True Outcomes", 
       ylab="T-statistic with Predicted Outcomes")

abline(0,1)

legend("topleft", col=c("orange", "light blue", "blue"), pch=15, 
       legend= c("No Correction", "Non-Parametric Bootstrap", "Parametric Bootstrap"))

title(main = "T-Statistics")

dev.off()

# P-value Plot
png(filename=paste0(covariate, "/P_Values.png"), units="in", width=6, height=6, res=300)

pval <- read.csv("pvals.csv")
pval$class <- as.factor(pval$class)

hextri(x=pval$x, y=pval$y, 
       class=pval$class, colour=c("orange", "lightblue", "blue"),
       style="size", nbins=25, 
       xlab="P-value with True Outcomes", 
       ylab="P-value with Predicted Outcomes")

abline(0,1)

legend("topleft", col=c("orange", "light blue", "blue"), pch=15, 
       legend= c("No Correction", "Non-Parametric Bootstrap", "Parametric Bootstrap"))

title(main = "P-Values")

dev.off()
