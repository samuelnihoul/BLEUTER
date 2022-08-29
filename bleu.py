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


    "If carbon dioxide hits a new high every year, why isn't every year hotter than the last?"
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