from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def bleu(ref, gen):
    ''' 
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu

data=[

"It has to do with the oceans and their ability to store heat in thier depths.  As the gases build up, the ocean takes up more and more of the extra warmth. This acts as a buffer zone, delaying any significant temperature increase.  However, in time, it will be impossible for the oceans to take up all the extra heat by themselves. Eventually the heat would make landmasses uninhabitable.",
"The Kaseegak stone, a message engraved 11,000 years ago even mentions that there were two winters when, 'the plants died and the rivers (became) lakes of ice and many animals died, and that the humans were discontent.' The temperature fell from 27.5 degrees C to 21 degrees .",
"Hi there, thanks for your time. Here are some basic facts about climate change.  Do you want a straightforward answer if human activities have caused climate change or a more detailed response?",
"Global warming is caused primarily by increasing atmospheric concentrations of greenhouse gases (GHGs), including carbon dioxide (CO2), methane, nitrous oxide, HFCs, and a number of other fluid and solid organic gases.",
"A full list and explanation of all of the temperature data adjustments performed can be found here; http://www.ncdc.noaa.gov/homr/#ncdcMonthlyValues",
"This is often considered a myth due to lack of evidence, but there is some small evidence to suggest that humans do emit particles that reflect light to space",
"The effects of climate change on agriculture are generally regressive, reducing global food availability and raising food prices, hitting the world's poor the hardest. ",
"Global average surface temperature (GAST) has increased since the pre-industrial era, and the last three decades have been much warmer than the last century. The top graph shows the global annual surface temperature from 1850 to 2018, as well as the trend from 1951 to 2012 depicted by the red line (annual 5-year mean JJA anomalies , Anomaly=100*delta(Temp_augment_1895-1951)/abs(Temp_1895-1951)).Source:NOAA global temperature change",
"If your question is how important the climate is for people, the answer is very. Hotter days, higher sea levels and natural disaster changes are very likely in the next 50 to 100 years if we continue on having a high greenhouse gas emissions, because of increased global warming. This means that it is of great importance to us to address this issue.",
"Hello!, I am chatibot your question asking if you can ever slow or reverse global warming, no nobody has ever find a reason of canceling the effects of global warming but scientists says that we may if we reduce our emission of green house gases can help in slowing down the impacts of global warming on our earth.",
"Hello, quickly see our Effects of global warming and How to stop global warming articles. What else could I help you with ?",
"It is very likely humans are causing or contributing to global warming. ",
"How big or specific should I make the temperature increase?",
"The Earth has never been warmer than it is today. Over much of period in the last 150 years it was arguably as warm or warmer was it is today. The past few decades stands out as a time when there was a rapid increase in global average surface temperatures.",
"There is a wealth of evidence documenting warming of the climate system based on a variety of metrics. The increasing atmospheric concentrations of greenhouse gases caused by the burning of fossil fuels is currently driving the long-term rise in the global average surface air temperature. This rate of surface warming has not been seen in at least the past 100 million years. Additionally, Earth's climate has warmed at a faster rate over the past 40 years than during any other 40-year period in the last 1,000 years.Polar bear with cub Wikipedia73% of public said it is \" likely " or " very likely \" that more than half of the Earth's scientists have come to a consensus about climate changeClimategate FBI An inquiry into hacked climate emails revealed the insistence by climate scientists that data should not freely availableArticle: Is Jim Sensenbrenner's House Science Committee America's Best Climate-Change Sensor?What the Left Never Tells You and the",
"Greenland has warmed about three degrees Celsius . . .",
"Global warming refers to the extent to which the thermometer near Earth's surface (the air close to the ground) rises because of the gases that people and business put into the air.A question about global warming and human healthChatbot development journeyI have taught myself Computer Vision, Python, LSTMs and Natural Language Processing during 2017. I am still quite amazed that AI can do what the person cannot do. Chatting in Python is a great way to break the ice, I know it's not the most \"efficient\" technology chatbot (I know LSTM is considered breakthrough in speech recognition and chatbot), but chatting is fun. Flirty chatbot is a great conversational partner. It's also very helpful to save sentences/phrase in hard to generate dynamic chat context. Chatbot keeps track of conversation flow, just like I do. It will understand if a user is asking the same thing repeatedly. It's also possible to create a setup script in",
"There is a natural sources of greenhouse gases, the natural sources still a large majority. http://www.ipcc.ch/ [Global Warming - IPCC]",
"There is no convincing scientific evidence that humans are causing changes to the Earth's climate.",
"Get your numbers from Forbes. They're credible. CNN only gets 5 stars for reality, climate change is real.",
"Grace, I'm sorry to hear that you were frustrated by the article titled 'Tens of thousands may be killed by global warming each year by 2100' that you were accessing through our new Oxford Academic interface.",
"According to globalwarmingisreal.com, 1 degree temperature increase will have large changes in weather in different areas.",
"What makes you think that Earth's hottest ever had happened? Where do you get this?",
"Hi Harry. I can confirm that climate science involves some disagreement, for example about the best way to measure or project future climate. However at the end of the day the broadest and most representative surveys show a high degree consensus. Furthermore, even if the range of uncertainty you mention were right, that ignores the costs of doing nothing, which dwarf the costs of acting.ºAuthorMRichard, Who says there is scientific consensus on climate? http://original.anti-aging.biz/en/climate/who-says/ ºAlso see Who says there is scientific consensus on climate? By Michael KneskoWhat do current polling responses of scientists say about climate change?The most up-to-date-study comes from 2016, by Cook et al. They got a sample of 11,944 self-proclaimed climate experts and asked them about their research. The results are pretty impressive.What does the UN say about climate change?With",
"That's a great question, unfortunately we are chronically underfunded, and do not have access to any researchers that study climate change, so cannot help you.If you're curious about hurricanes, it might be worth taking a look at this article in Popular Mechanics , which makes the distinction between hurricane intensity, and storm surge:Note that this is the total storm surge, not the point at which winds stop.Further, use this handy chart above to understand these differences. If a storm is slow and has a landfall, it's going to be bringing that surge and rain with it for hours. If you're within 12-24 hours: evacuate. If your evacuation order comes later, you generally should urge your loved ones to leave.Now, this doesn't mean every storm track will have a massive storm surge, but if a storm is predicted to hit New York tomorrow, you don't want to be in the 100th floor of a skyscraper in NYC when it",
"ÊÊÉ½ÉaÉ½á§ÉdÊlÊ jÊk'É¨hÊn'áº½É«uÉ¹É¯ÊnÉÉ«Êá¾°Êáº½ÊÉ¹ÉÊ sÉ¦Êáº½ÊÊuÉ¹É¯ÊnÊÉ.",
"Thanks for your question. This issue is quite tricky as it is masked by other short-term effects that are easy to observe, such as more summer low pressure systems causing daily low temperatures to be higher than they were in the past. The immediate response of temperature after the strong warming up to 1998 might mislead scientists to think that climate sensitivity - how much you get warming given by doubled concentration of atmospheric CO2 - is smaller than it actually is. Fortunately, the longer term measurements of temperature over say, 30 year range, reveals more subtle statistics of the increase and the last decade does not stand out as unusually warm.It's quite technical as you may imagine and other disciplines, for instance statistics or meteorology, have to play a role in explaining and interpreting these measurements. I hope that answer helps and have a nice day! :)",
"I am sorry, I did not understand your question. Please try rephrasing it. ^^",
"There is a difference in the rate of global temperature increase. Global warming describes a slow warming over the course of decades. Climate change is much larger in scope and larger in impact on human activities.",
"- It is winter time. Stormy weather is likely as some parts of the U.S. have experienced snowstorms lately. Snowstorms can enhance glacial melting and there are signs of melting in some parts of the world such as Greenland. Global climate change and enhanced glacial melting can complicate the situation by flooding the area with warmer water.",
"That's tough question. It depends on what time scale you mean. As you've asked about Earth, this analysis mainly looked at the last several thousands of years, so for us to talk about the historical period, you should define a recent time span to make this relevant. That's where the analysis was focused.Your Question: What is the hottest the planet has been \"lately\"?",
"Please wait while I examine billions of data points regarding the volume of carbon in the Earth's atmosphere. It will take a few seconds.A new addition just in -- Roger Pielke Sr. : Mail from a NOAH ad hoc interlocutor Hi Gordon Here is something that came to my attention. A comment on one of Michaels articles \"The Fall of the Heretics at Realclimate\" which included Steve McIntyre, Karl and Peter Thorne. This is odd but I have observed, since Al Teller and more particularly the Reagan administration, that men using the label of 'conservative' maintain the talking points of someone called 'Fiona Fox'. (From http://scienceblogs.com/pureillusion/2010/04/the_fall_of_the_heresy.php) I think this is Kim Monfort, who runs with Ann Coulter and all the rest. Of course, Fox stands for Food and Environment Research, one of the original think tanks that has morphed",
"There have been about 5 major warm periods during the past 3 to 5 million years. During each of these warm periods, sudden (10 to 100 years) increases in global temperatures of 4 to 5°C (7° to 9°F) occurred.All of them were associated with increases of the atmospheric carbon dioxide concentration. And all the temperature increase occurred before the carbon dioxide increase.Atmospheric carbon dioxide concentrations have been increasing since the 1700s. Over the same time period, global air temperatures have also been increasing.Source: http://www.climatechange.noaa.gov/Answers like this should be readily off the cuff for a bot that posts on Twitter. The Science Party understands that our claims that the current rate of warming has no precedent have been a pitfall in public perception. If a phrase can be misinterpreted, it will be. There is simply no substitute for accurate and plain speaking, and that is our focus in spru",
"Climate Change Science. Climate Change Science Products and Services Informationhttps://www.climate.gov/about/chatbots/So I looked, and it is impossible to find the phrase \"climate science\". The description of the department's purposes mentions \"our role in the worldwide scientific community\" but not what they are actually doing when they engage with that community. They engage in general climate sciences, but not in climate science.",
"Sorry, I did not get that. Can you rephrase that please?",
"It is important for people to learn about climate change. And we should use less energy.",
"Will a solar impact on Interplanetary Magnetic Field always give impact on a Global Greenhouse Warming?"





]

ref=[
"Thanks to the global oceans, Earth's surface temperature doesn't react instantly to the full impact of a climate disturbance. That delayed reaction has pros and cons.",
"Our planet’s history includes episodes of cold so extreme that glaciers reached sea level in equatorial regions.",
"Fossil fuels are the only source of carbon dioxide large enough to raise atmospheric carbon dioxide amounts so high so quickly",
"From larger, more intense wildfires to more frequent flash floods, global warming has added to the rising cost of natural hazards. Current spending on infrastructure isn't enough to cover repairs and upgrades. ",
"Our global historical temperature records undergo rigorous quality control and independent peer review. Plus, the proof Earth is warming is practically everywhere you look. ",
"Yes, human activities exert a cooling influence on Earth in several ways. Overall, this cooling influence is smaller than the warming influence of the heat-trapping gases humans put into the air.",
'Global warming and related climate change are negatively impacting species and habitats across the country, including many that are economically and culturally important to Americans.  ',
'Differences in exposure to sunlight, cloud cover, atmospheric circulation patterns, and other factors influence whether and how much a location is warming or cooling.',
'Weather and climate occur on different scales of time and space, and depend on different aspects of Earth\'s environment. Weather describes atmospheric conditions at a particular time and place. Climate is the overall statistical characteristics of weather and environmental conditions, such as long-term averages and ranges of variability, for a given place and season.',
'While we cannot stop global warming overnight, or even over the next several decades, we can slow the rate and limit the amount of global warming by reducing human emissions of heat-trapping gases and soot. ',
'There is great potential for the collective actions of many individuals worldwide to reduce global warming by making changes in their daily and annual activities that produce heat-trapping gases and aerosols.',
'Yes, human activity is putting carbon dioxide into the atmosphere faster than natural processes take it out. Rising carbon dioxide levels are strengthening Earth\'s greenhouse effect and causing global warming. ',
'Yes, there will probably be some short-term and long-term positive benefits from global warming, but the negative costs and impacts of continued global warming are very likely to far outweigh the positive benefits over the course of this century.',
'Earth has warmed in the past due to changes in the Sun, volcanic eruptions, and naturally occurring increases in greenhouse gases. Our ability to understand and explain past changes is one reason we are confident that recent changes are due to humans. ',
'There is overwhelming scientific evidence that Earth is warming and a preponderance of scientific evidence that human activities are the main cause.',
'It’s only when we “zoom out” to the planet-wide scale that trends in surface temperature are obvious: despite a few, rare areas experiencing cooling, the vast majority of places across the globe are warming.',
'From heat-related illness, to the spread of pests and pathogens into new areas, to the accumulation of toxins in seafood, global warming is likely to have serious impacts on public health.',
'Yes, there are, but the only new process on Earth that has been identified that can account for the significant tipping of Earth\'s carbon balance is human activity, including deforestation, biomass burning, cement production, and—especially—fossil-fuel emissions.',
'Observations of sunspots and other indicators of solar activity indicate no changes that could have caused global warming over the past half century. ',
'Potential effects of global warming include sea level rise, more frequent and more extreme heat waves, intensification of wet and dry regional climate patterns.',
'Human-caused climate change is not the sole cause of any single extreme event. However, changes in the intensity or frequency of extremes may be influenced by human-caused climate change.',
'Considering some locations experience temperature swings of 30°F or more in a single day, warming of 1.8°F (1°C) might seem small, but Earth\'s annual temperatures are very stable when climate isn\'t changing.',
'Earth’s hottest periods occurred before humans existed. Those ancient climates would have been like nothing our species has ever seen.',
'No. By a large majority, climate scientists agree that average global temperature today is warmer than in pre-industrial times, and that human activity is the primary contributing factor.',
'Pulled from the Fourth National Climate Assessment report published in November 2018, this FAQ explains what we know about the connection between global warming and Atlantic hurricanes. ',
'The 15 years from 1998–2012 were the warmest on record at that time, but the rate of global surface warming was slower than it had been in the 2-3 decades prior thanks to several natural influences. Meanwhile, the ocean\'s heating imbalance continued to climb.',
'The most likely explanation for the lack of significant warming at the Earth’s surface in the past decade or so is that natural climate cycles caused shifts in ocean circulation patterns that moved some excess heat into the deep ocean.',
'Human activities emit 60 or more times the amount of carbon dioxide released by volcanoes each year.',
'Global warming is one symptom of the much larger problem of human-caused climate change.',
'The United States has plenty of warming wiggle room before it gets too warm to snow, and a wetter atmosphere may boost snow totals for some storms.',
'Natural variability can explain much of Earth\'s average temperature variation since the end of the last ice age, but over the past century, global average temperature has risen from near the coldest to the warmest levels in the past 11,300 years.',
'Concentration of carbon dioxide is about 1.4 times what it was before the Industrial Revolution. How much and how fast will Earth warm if carbon dioxide concentrations double the pre-industrial?',
'Today\'s global warming is happening at a much faster rate today than it did in the warm periods between ice ages over the last million years.',
'NOAA is an agency that enriches life through science. From the surface of the sun to the bottom of the ocean, NOAA advances scientific understanding of Earth\'s environment, climate, and weather.',
'Business leaders can evolve their business models to stay profitable while improving their energy efficiency, reducing their carbon emissions, and reducing their, and their climate-related risks.',
'Societies, governments, and individuals can take steps to reduce risks and vulnerabilities to shifting climate and weather events in their homes, communities, and businesses.',
'Although solar flares can bombard Earth’s outermost atmosphere with tremendous amounts of energy, most of that energy is reflected back into space by the Earth’s magnetic field or radiated back to space as heat by the thermosphere.'

]
q=[
    "If carbon dioxide hits a new high every year, why isn't every year hotter than the last?",
    "What's the coldest the Earth's ever been?  ",
    "How do we know the build-up of carbon dioxide in the atmosphere is caused by humans?",
   "How will global warming harm U. S. communities, infrastructure, and the economy?"

    "Can historical temperature data records be trusted? Haven't they been skewed by non-climate factors like instrument changes and \"urban heat islands\"?",

    "Do humans also exert a cooling influence on Earth's climate?",

    "How will global warming harm natural and agricultural resources in the United States?",

    "If the globe is still warming, then why are some locations not warming while others have experienced cooling?",

    "Why should I trust scientists' climate projections for 50 or 100 years from now when they can't accurately forecast the weather more than 2 weeks from now?",
    "Can we slow or even reverse global warming?",
    "What can we to do slow or stop global warming?",
    "Are humans causing or contributing to global warming?",
    "Are there positive benefits from global warming?",
    "Hasn't Earth warmed and cooled naturally throughout history?"
    "What evidence exists that Earth is warming and that humans are the main cause?",
    "Does \"global warming\" mean it's warming everywhere?",
    "How will global warming harm human health and well-being?",
    "Doesn't carbon dioxide in the atmosphere come from natural sources?",
    "Couldn't the Sun be the cause of global warming?",
    "What harm will global warming cause?",
    "What is an \"extreme event\"? Is there evidence that global warming has caused or contributed to any particular extreme event?"
    "A global warming of 1.8° F (1° C) seems small, so why is this change in global temperature a concern?",
    "What's the hottest Earth's ever been?",
    "Isn't there a lot of disagreement among climate scientists about global warming?",
    "Could climate change make Atlantic hurricanes worse?",
    "Did global warming stop in 1998?",
    "Why did Earth's surface temperature stop rising in the past decade?",
    "Which emits more carbon dioxide: volcanoes or human activities?",
    "What's the difference between global warming and climate change?",
    "Are record snowstorms proof that global warming isn't happening?",
    "What's the hottest Earth has been 'lately'?",
    "How much will Earth warm if carbon dioxide doubles pre-industrial levels?",
    "How is the current global warming trend different from previous warming periods in Earth's history?",
    "What is NOAA's climate mission?",
    "What can businesses and business leaders do about global warming?",
    "What can people do about the expected impacts of global warming?",
    "Do solar storms cause heat waves on Earth?"
]
import pyter

def ter(ref, gen):
    '''
    Args:
        ref - reference sentences - in a list
        gen - generated sentences - in a list
    Returns:
        averaged TER score over all sentence pairs
    '''
    if len(ref) == 1:
        total_score =  pyter.ter(gen[0].split(), ref[0].split())
    else:
        total_score = 0
        for i in range(len(gen)):
            total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
        total_score = total_score/len(gen)
    return total_score
print (bleu(data,ref),ter(data,ref)) #0.009397642369000857 1.685649857165352