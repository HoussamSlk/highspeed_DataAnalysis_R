##Group members:
## Kevin Hofman, u376673
## Husam Alsalek, u600968
## Osama Soumakie, u473331
##
## Load packages ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

## Load data -------------------------------------------------------------------
airbag <- read.delim("../input/nassCDS.csv", stringsAsFactors = FALSE , sep = ',')

## Extracting highspeed accidents
highSpeed <- airbag[airbag$dvcat=="55+",]

##creating a safety system feature to indicate which safety system the vehicle 
## has: None, airbag, seatbeat, or airbag & seatbelt 
highSpeed$safety_system <- 0
highSpeed$safety_system[highSpeed$airbag == "none" & 
                        highSpeed$seatbelt == "none"]<-"none"
highSpeed$safety_system[highSpeed$airbag == "airbag" & 
                        highSpeed$seatbelt == "none"]<-"AB"
highSpeed$safety_system[highSpeed$airbag == "none" & 
                        highSpeed$seatbelt == "belted"] <-"SB"
highSpeed$safety_system[highSpeed$airbag == "airbag" & 
                        highSpeed$seatbelt == "belted"] <-"ABSB"
##-----------------------------------------------------------------------------------------------
##graph demonstrating the relation between safety systems and injury severity
counts<-table(highSpeed$injSeverity,highSpeed$safety_system)
safetyGraph <- barplot(counts,main = "Safety System VS Severity of Injury",
                       xlab ="Safety system",ylab = "Frequency",
                       col = c("#733080","#F4BAC8","#A40607","#7288B9", 
                               "#F0C595","#000000"),
                       legend=c("None", "Possible Injury", "No Incapacity",
                                "Incapacity","Killed","Unkown",
                                "Prior Death"),beside = TRUE,
                       args.legend = list(x="topleft"))
                       
##graph demonstrating the relation between safety systems and survival rate
death_alive_count <-table(highSpeed$dead,highSpeed$safety_system)
deathGraph <- barplot(death_alive_count,main = "Safety System VS Survival",
                      xlab ="Safety system",ylab = "Survival",
                       col = c("#808080","#FFFFFF"),
                      legend=rownames(death_alive_count),beside = TRUE, args.legend = list(x="topleft"))


##Predicting survival rate from safety system
set.seed(1)
trn_index = createDataPartition(y = highSpeed$dead, p = 0.70, list = FALSE)
trn_highspeed = highSpeed[trn_index, ]
tst_highspeed = highSpeed[-trn_index, ]
### K-Nearest Neighbors
set.seed(1)
survival_knn = train(dead ~ safety_system, method = "knn", data = trn_highspeed,
                   trControl = trainControl(method = 'cv', number = 5))
summary(survival_knn)
str(survival_knn)

## Logestic regression 
set.seed(1)
trn_highSpeed_r <- highSpeed
## re-level 'dead' so that 'alive' is the reference level, indicating survival
trn_highSpeed_r <- trn_highSpeed_r %>%
  mutate(dead = relevel(factor(dead),
                                  ref = "alive"))
## fit the logistic regression model
survival_lgr = train(dead ~ safety_system, method = "glm",
                    family = binomial(link = "logit"), data = trn_highSpeed_r,
                    trControl = trainControl(method = 'cv', number = 5))

summary(survival_lgr)
str(survival_lgr)

max(survival_knn$results$Accuracy)
max(survival_lgr$results$Accuracy)


## using LVQ model to estimate the importance of features
# ensure results are repeatable
set.seed(1)
# load the library
library(mlbench)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
highSpeed_lvq <- highSpeed
#changing 'dead', 'alive' to numerical values: 0 , 1
highSpeed_lvq$dead[highSpeed_lvq$dead=="dead"] <- 0
highSpeed_lvq$dead[highSpeed_lvq$dead=="alive"] <- 1
model <- train(dead~safety_system, data=highSpeed_lvq, method="lvq", 
               preProcess="scale", trControl=control)
max(model$results$Accuracy)
