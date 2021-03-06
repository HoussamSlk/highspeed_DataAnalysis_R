# Highspeed_DataAnalysis_R
Predicting survival rates based on the use of airbags and seat belts in high-speed car accidents dataset
High-speed traffic accidents: can we predict survival rates based on the use of airbags and seat belts?


Done by: Kevin Hofman,Husam Alsalek & Osama Soumakie
Introduction

Every year, traffic accidents cause around 1.35 million deaths worldwide (World Health Organisation, 2018). Risk factors for traffic injury or death include speeding, driving under the influence of alcohol or drugs, unsafe roads, distracted driving, etc. Additionally, the World Health Organisation mentioned that traffic accidents cause economic losses, which result from for example decreased productivity caused by disability or death and the cost of treating injuries. The above has made traffic accidents a complex problem concerning governments, car manufacturers, and people in general. 
In the pursuit of overcoming this issue, much effort has been invested in making vehicles safer, for instance through the development of seat belts and airbags. Support for the added safety provided by airbags has been found in a study by Intas and Stergiannis (2011), wherein they reported a reduction of passenger injuries of up to 50%. Regarding the effects of seat belts, Abbas, Hefny, and Abu-Zidan (2011) mentioned a highly significant negative correlation between traffic deaths and proper use of seat belts. Even though the introduction of these vehicle safety devices has reduced the dangers involved in car crashes, the problem of traffic accident fatalities is far from being solved. An unintended consequence of both airbags and seat belts is that they could also injure the occupants of the vehicle. For example, Abbas, Hefny, and Abu-Zidan (2011) mentioned that incorrect use of seatbelts could even cause fatal injuries. Airbags on the other hand might cause chemical or thermal burn injuries, and the deployment could cause blunt trauma (Suhr & Kreusch, 2004). 
Taking into consideration the pros and cons of airbags and seat belts, this study aims to investigate the added effectiveness of combining both safety measures. Specifically of interest is whether this effect on protection from fatal injury is significantly higher than solely using either seat belts or airbags by itself when a traffic accident occurs at speeds over 55 kilometers per hour. Further comparison will be made with accidents wherein neither safety measure was utilized in order to establish a baseline. Expected is that the protection provided by using both seat belts and airbags combined will outweigh the benefits of each measure separately when one is involved in a high-speed car accident.


Methods

To assess our research question, the NASS CDS database (National Automotive Sampling System - Crashworthiness Data System, n.d.) has been used. This dataset consists of 26217 observations and 17 variables concerning car accidents which involved injury or damage to property, whereby at least one of the vehicles involved was towed. The data has been acquired and assembled through collecting police reports in the United States between 1997 and 2002. The people included in the data are limited to those situated in the front seats of the vehicle at the time of the accident. 
	Since the dataset was tidy and correctly structured from the onset, preprocessing was not necessary aside from simple filtering operations. For this study, the observations in the dataset were filtered to only include accidents where a speed of 55 kilometers per hour or more has been reported. To assist in the analysis and interpretation of the data, an additional variable was added to divide the data into different degrees of use of safety measures. These groups were: accidents with only seat belts protecting the occupants, those protected by only airbags, those protected by both, and accidents were neither seat belts nor airbags were utilized, consisting of 251, 393, 360, and 488 observations, respectively.
Finally, logistic regression, K-nearest neighbor, and learning vector quantization were applied to the dataset to investigate whether survival rates could be predicted based on the safety system used.


Results
When comparing the survival rates accompanying the different safety measures, as presented in appendix A1, the current data suggests that the survival rates are lowest for the airbag only group, followed by the group with no protective measures, the group with only seatbelts, and finally the group with both an airbag and seatbelt. To further illustrate the effects of airbags and seat belts, appendix A2 shows the frequency of different levels of injury occurring per group of safety measures. Additionally, appendix A3 shows whether those involved in an accident survived the crash or not, again grouped by safety measures taken.
The insights gained regarding the survival rates for each group are further expanded by the results of the logistic regression, K-nearest neighbor, and learning vector quantization. The maximum accuracy when predicting the survival rate from the used safety measures was found to be 72% on average. See Appendix B1 for the accuracy for each technique separately. 

Discussion
Results from the current study indicate that using both an airbag and a seat belt provides the most protection, as this group had the highest survival rate. This is in line with the initial expectation that the combined use of an airbag and a seat belt provides the most protection against fatal injuries when one is involved in a traffic accident at speeds over 55 kilometers per hour. These findings furthermore are in line with earlier research by Intas and Stergiannis (2011) and Abbas, Hefny, and Abu-Zidan (2011), as they also found significant protective benefits of airbags and seat belts, respectively. Another interesting result of the data analysis is that the occupants who were only protected by an airbag appear to have the lowest survival rate of all examined groups. This may suggest that airbags should not be treated as an independent safety measure, but as a supplement to seat belts to further increase protection against fatal injuries.
In sum, the results of the current study are in line with previous expectations, confirming the hypothesis that seat belts and airbags are more effective in protecting against fatal injuries when both are used together, compared to using no protection, or only seat belts or airbags separately.
Limitations
When interpreting the results of the current study, one should keep several limitations in mind. First, the dataset solely consisted of data gathered in the United States. Since factors like road quality or the degree of traffic law enforcement can differ widely between countries, one should be cautious when generalizing the data to countries other than the United States.
Additionally, the dataset contains observations where an airbag was present but was not deployed. As there is no data available on the reasons and consequences of the airbag not deploying, this might have an effect on the results. Specifically, the survival rates in the airbag- and airbag plus seatbelt groups could be affected by faulty deployment or non-deployment of airbags. Further research could investigate this factor to determine whether the survival rates in the airbag groups, or other groups for that matter, are significantly skewed as a result.







References
Abbas, A. K., Hefny, A. F., & Abu-Zidan, F. M. (2011). Seatbelts and road traffic collision injuries. World journal of emergency surgery, 6(1), 18.
Intas, G., & Stergiannis, P. (2011). How safe are the airbags? A review of literature. Health Science Journal, 5(4), 262.
National Automotive Sampling System - Crashworthiness Data System (NASS-CDS) - NASS-CDS.(n.d.). Airbag and other influences on accident fatalities. Retrieved from https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nassCDS.html
Suhr, M., & Kreusch, T. (2004). Burn injuries resulting from (accidental) airbag inflation. Journal of Cranio-Maxillofacial Surgery, 32(1), 35-37.
World Health Organization. (2018). Road Traffic Injuries. https://www.who.int/en/news-room/fact-sheets/detail/road-traffic-injuries
