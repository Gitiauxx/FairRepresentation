options(tz="CA")
require(tikzDevice)

# figure 1 a
res_fig <- read.csv("C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\synth_exp_violation_delta_test3.csv")
res_fig <- data.table(res_fig)
res_fig[, estimated:=estimated_delta]
res_fig[, size:="5K"]
res_fig[, low:=estimated - delta_deviation]
res_fig[, high:=estimated + delta_deviation]

res_fig2 <- read.csv("C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\synth_exp_violation_delta_1K.csv")
res_fig2 <- data.table(res_fig2)
res_fig2[, estimated:=estimated_delta]
res_fig2[, size:="1K"]
res_fig2[, low:=estimated - delta_deviation]
res_fig2[, high:=estimated + delta_deviation]

res_fig <- rbind(res_fig, res_fig2)

tikz(file = "figure1a.tex", width = 1.75, height = 1.75)

plt1 <- ggplot(res_fig)
plt1 <- plt1 + geom_point(aes(x=delta, y=estimated, color=size, shape=size), show.legend = F)
plt1 <- plt1 + geom_line(aes(x=delta, y=estimated, color=size))
plt1 <- plt1 + geom_ribbon(aes(x=delta, ymin=low, ymax=high, fill=size), alpha=0.2, show.legend=F )
                  
plt1 <- plt1 + scale_fill_manual(values = c("5K"="red", "1K"="blue"))
plt1 <- plt1 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                 panel.background = element_blank(), axis.line = element_line(colour = "black"))
plt1 <- plt1 +  scale_color_manual(values=c("5K"="red", "1K"="blue"), labels=c("5K", "1K"), breaks=c("5K", "1K"))
plt1 <- plt1 + geom_abline(slope=1, intercept=0)
plt1 <- plt1 + annotate(geom="text", x=1, y=2.5, label="Over-estimate",
                              color="black", size=2.75, angle=45)
plt1 <- plt1 + annotate(geom="text", x=1.75, y=0.75, label="Under-estimate",
                        color="black", size=2.75, angle=45)
plt1 <- plt1 + scale_x_continuous(limits = c(0, 3.5))
plt1 <- plt1 + scale_y_continuous(limits = c(0, 3.5))
#plt1 <- plt1 + theme(legend.position="bottom")

plt1 <- plt1 + theme(text = element_text(size=17))
plt1 <- plt1  + theme(legend.position = c(0.9, 0.3), legend.text=element_text(size=6))
plt1 <- plt1 + theme(legend.key=element_blank(),
                 legend.title=element_blank(),
                 legend.box="vertical")
plt1 <- plt1 + theme(text = element_text(size=10))
plt1 <- plt1 + guides(color=guide_legend(
  keywidth=0.1,
  keyheight=0.1,
  default.unit="inch")
)

#plt1 <- plt1  + theme(legend.position=c(0, 3))
plt1 <- plt1 + labs(x="True $\\delta_{m}$", y="Estimated $\\hat{\\delta}_{m}$")
plt1

dev.off()


res_ad <- read.csv("C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\worst_violations_auditors.csv")
res_ad <- data.table(res_ad)
res_ad[, estimated:=estimated_delta]
res_ad[, low:=estimated - delta_deviation]
res_ad[, high:=estimated + delta_deviation]


tikz(file = "figure1b.tex", width = 1.75, height = 1.75)
plt3 <- ggplot(res_ad)
plt3 <- plt3 + geom_point(aes(x=delta, y=estimated, color=auditor, shape=auditor), show.legend = F)
plt3 <- plt3 + geom_line(aes(x=delta, y=estimated, color=auditor))

plt3 <- plt3 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                     panel.background = element_blank(), axis.line = element_line(colour = "black"))
plt3 <- plt3 +  scale_color_manual(values=c("dt"="red", "rf"="blue", "svm_linear"="gray", "svm_rbf"="brown"), 
                                   labels=c("DT", "RF", "SVM-Lin", "SVM-RBF"), 
                                   breaks=c("dt", "rf", "svm_linear", "svm_rbf"))
plt3 <- plt3 + geom_abline(slope=1, intercept=0)
plt3 <- plt3 + annotate(geom="text", x=1, y=2.5, label="Over-estimate",
                        color="black", size=2.75, angle=45)
plt3 <- plt3 + annotate(geom="text", x=1.75, y=0.75, label="Under-estimate",
                        color="black", size=2.75, angle=45)
plt3 <- plt3 + scale_x_continuous(limits = c(0, 3.5))
plt3 <- plt3 + scale_y_continuous(limits = c(0, 3.5))
#plt1 <- plt1 + theme(legend.position="bottom")

plt3 <- plt3 + theme(text = element_text(size=17))
plt3 <- plt3  + theme(legend.position = c(0.875, 0.15), legend.text=element_text(size=6))
plt3 <- plt3 + guides(color=guide_legend(
            keywidth=0.09,
            keyheight=0.1,
            default.unit="inch")
            )
plt3 <- plt3 + theme(legend.key=element_blank(),
                     legend.title=element_blank(),
                     legend.box="vertical")
plt3 <- plt3 + theme(text = element_text(size=10))
#plt1 <- plt1  + theme(legend.position=c(0, 3))
plt3 <- plt3 + labs(x="True $\\delta_{m}$", y="Estimated $\\hat{\\delta}_{m}$")
plt3

dev.off()


# figure 1c: effect of unbalance
res_bal <- read.csv("C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\synth_exp_unbalance_3a.csv")
res_bal <- data.table(res_bal)
res_bal <- res_bal[balancing%in%c("Uniform", "IS", "MMD_NET")]
res_bal[balancing == 'MMD_NET', balancing:="MMD"]

tikz(file = "figure1c.tex", width = 1.75, height = 1.75)

plt2 <- ggplot(res_bal, aes(x=unbalance, y=bias, color=balancing, shape=balancing))
plt2 <- plt2 + geom_point(show.legend = F)
plt2 <- plt2 + geom_line()
plt2 <- plt2 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                     panel.background = element_blank(), axis.line = element_line(colour = "black"))
plt2 <- plt2 +  scale_color_manual(values=c("Uniform"="red", "IS"="blue", "MMD"="grey"), 
                                   labels=c("UW", "IS", "MMD"), breaks=c("Uniform", "IS", "MMD"))
plt2 <- plt2 + scale_y_continuous(labels = function(x) format(x, scientific = TRUE))

#plt1 <- plt1 + theme(legend.position="bottom")
plt2 <- plt2 + theme(text = element_text(size=17))
plt2 <- plt2  + theme(legend.position = c(0.35, 0.75), legend.text=element_text(size=6))
plt2 <- plt2 + theme(legend.key=element_blank(),
                     legend.title=element_blank(),
                     legend.box="vertical")
plt2 <- plt2 + theme(text = element_text(size=10))
plt2 <- plt2 + labs(x="Unbalance  $\\mu$", y="Bias $\\hat{\\gamma}-\\gamma$")
plt2

dev.off()

# figure 1c: effect of unbalance
res_iter <- read.csv("C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\worst_violations_iteration3.csv")
res_iter <- data.table(res_iter)
res_iter[, gamma:=log(gamma)]
res_iter <- res_iter[step == 0.01]
res_iter <- res_iter[nu > 15]


tikz(file = "figure1d.tex", width = 1.75, height = 1.75)

plt4 <- ggplot(res_iter)
plt4 <- plt4 + geom_point(aes(x=nu, y=alpha, color="alpha"), shape=17, show.legend = F)
plt4 <- plt4 + geom_line(aes(x=nu, y=gamma, color="alpha"))
plt4 <- plt4 + geom_point(aes(x=nu, y=gamma, color="gamma"), shape=18, show.legend = F)
plt4 <- plt4 + geom_line(aes(x=nu, y=gamma, color="gamma"))
plt4 <- plt4 + geom_point(aes(x=nu, y=delta, color="delta"), shape=19, show.legend = F)
plt4 <- plt4 + geom_line(aes(x=nu, y=delta, color="delta"))
plt4 <- plt4 + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                     panel.background = element_blank(), axis.line = element_line(colour = "black"))
plt4 <- plt4 +  scale_color_manual(values=c("alpha"="red", "gamma"="blue", "delta"="black"), 
                                   labels=c("size", "$\\hat{\\delta}_{m}$", "$\\delta_{m}$"), breaks=c("alpha", "gamma", "delta"))
plt4 <- plt4 + scale_y_continuous(labels = function(x) format(x, scientific = TRUE))

#plt1 <- plt1 + theme(legend.position="bottom")
plt4 <- plt4 + theme(text = element_text(size=17))
plt4 <- plt4  + theme(legend.position = c(0.25, 0.7), legend.text=element_text(size=6))
plt4 <- plt4 + theme(legend.key=element_blank(),
                     legend.title=element_blank(),
                     legend.box="vertical")
plt4 <- plt4 + theme(text = element_text(size=10))
plt4 <- plt4 + labs(x="Iterations", y="Estimates")
plt4 <- plt4 + guides(color=guide_legend(
  keywidth=0.09,
  keyheight=0.1,
  default.unit="inch")
)
plt4

dev.off()
