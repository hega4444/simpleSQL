import tiktoken

import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = """
Here's an extract from the article:
{
    "nid": "nid_0013",
    "title": "Israelis overwhelmingly are confident in the justice of the Gaza war, even as world sentiment sours",
    "text": "FILE - People demonstrate alongside relatives of people kidnapped during the Oct. 7 Hamas cross-border attack in Israel, during a protest calling for the return of the hostages, in Tel Aviv, Israel, Saturday, Nov. 4, 2023. At a time when world sentiment has begun to sour on Israel's devastating airstrikes in Gaza, the vast majority of Israelis, across the political spectrum, are convinced of the justice of the war. (AP Photo/Bernat Armangue, File)\n\nFILE - People demonstrate alongside relatives of people kidnapped during the Oct. 7 Hamas cross-border attack in Israel, during a protest calling for the return of the hostages, in Tel Aviv, Israel, Saturday, Nov. 4, 2023. At a time when world sentiment has begun to sour on Israel's devastating airstrikes in Gaza, the vast majority of Israelis, across the political spectrum, are convinced of the justice of the war. (AP Photo/Bernat Armangue, File)\n\nFILE - People demonstrate alongside relatives of people kidnapped during the Oct. 7 Hamas cross-border attack in Israel, during a protest calling for the return of the hostages, in Tel Aviv, Israel, Saturday, Nov. 4, 2023. At a time when world sentiment has begun to sour on Israel's devastating airstrikes in Gaza, the vast majority of Israelis, across the political spectrum, are convinced of the justice of the war. (AP Photo/Bernat Armangue, File)\n\nFILE - People demonstrate alongside relatives of people kidnapped during the Oct. 7 Hamas cross-border attack in Israel, during a protest calling for the return of the hostages, in Tel Aviv, Israel, Saturday, Nov. 4, 2023. At a time when world sentiment has begun to sour on Israel's devastating airstrikes in Gaza, the vast majority of Israelis, across the political spectrum, are convinced of the justice of the war. (AP Photo/Bernat Armangue, File)\n\nAt a time when world sentiment has begun to sour on Israel\u2019s devastating offensive in Gaza, the vast majority of Israelis are convinced of the justice of the war\n\nJERUSALEM -- At a time when world sentiment has begun to sour on Israel's devastating offensive in Gaza, the vast majority of Israelis, across the political spectrum, are convinced of the justice of the war.\n\nStill under rocket and missile attacks on several fronts, they have little tolerance for anyone railing against the steep toll the conflict has exacted on the other side. They have rallied to crush Hamas, which breached the country\u2019s borders from the Gaza Strip, killing more than 1,400 people and taking over 240 hostage in an Oct. 7 rampage that triggered the war.\n\nCapturing the prevailing sentiment in Israel, former Prime Minister Ehud Barak said other countries would have reacted the same way to such a cross-border attack with mass casualties.\n\n\u201cThe United States would do whatever it takes,\u201d Barak recently told the magazine Foreign Policy. \u201cThey would not ask questions about proportionality or anything else.\u201d\n\nIsrael has carried out weeks of relentless airstrikes and launched a ground operation in what it says is a mission to destroy Hamas. More than 10,000 Palestinians have been killed in the fighting, according to the Health Ministry in Hamas-ruled Gaza.\n\nEntire neighborhoods have been flattened, more than half of the enclave\u2019s 2.3 million people have fled their homes, and food, water, fuel and medical supplies have dwindled dangerously under an Israeli siege.\n\nTo be sure, Palestinian citizens of Israel on the whole sympathize with the plight of the people of Gaza, while relatives of some hostages have expressed concern about what the bombing campaign means for their loved ones.\n\nBut since Oct. 7, the acrimonious polarization that had gripped Israel over Prime Minister Benjamin Netanyahu\u2019s proposal to weaken the country\u2019s courts has largely been replaced with an outburst of national unity. Some 360,000 Israeli reservists have been called up for a war that enjoys broad support, despite fears it will exact a high military toll. An estimated 250,000 people have been displaced by the violence.\n\nIsraelis are hanging the blue and white national flag on homes and cars, turning out in throngs to support hostage families, and handing out food at road junctions to soldiers headed to the front.\n\nTV stations broadcast under the slogans, \u201cIsrael at war,\" and \"Together we will win.\u201d A month after the attack, coverage focuses heavily on stories of grief and heroism, with little mention of the situation in Gaza.\n\nBacking for the war effort is pouring in from the home front as the government, caught flat-footed by the attack and distracted by infighting, struggles to meet vast new needs. From blood drives to food banks, volunteers have stepped in. One organization, HaShomer HaChadash, is helping to build bomb shelters, patrol farmlands in border areas, and keep farms going when their workers have been called up.\n\nIsraelis overwhelmingly are incensed by growing pro-Palestinian protests across the world \u2014 including within their own Palestinian community \u2014 and what they see as the demonization of Israel over the soaring Palestinian casualties. A global spike in antisemitic attacks has only deepened their commitment to a Jewish homeland.\n\n\u201cLet them put themselves in our shoes, with unending rocket fire on civilians for years,\u201d said Yosi Schnaider. Four of his relatives, including two young children, are hostages in Gaza. Two others were killed in the Hamas onslaught.\n\n\u201cThey\u2019ve been firing on Israel for years, carrying out attacks for years, and (Hamas\u2019) charter says its objective is to destroy Israel and the Jewish entity. What country would put up with that? I invite anyone who opposes (the war) to come here for a week. Then let\u2019s talk.\u201d\n\nWhile Israel initially was greeted with international sympathy in the first days after the attack, the humanitarian crisis in Gaza has drawn calls for a respite in the fighting, including from Israel\u2019s staunchest supporter, U.S. President Joe Biden. Bolivia severed diplomatic ties, and Jordan, Turkey, Chile and Colombia recalled ambassadors.\n\nThe ongoing violence has refocused world attention on the Palestinian struggle against more than half a century of Israeli military occupation and its stranglehold on the 5.5 million Palestinians living in east Jerusalem, the West Bank and Gaza. The last serious peace efforts broke down over a decade ago, and Netanyahu\u2019s government adamantly opposes Palestinian statehood.\n\nAt the same time, the fighting has shattered the illusion held by many in Israel that Palestinians could be sidelined because other countries in the region \u2014 the United Arab Emirates, Bahrain, Morocco and potentially Saudi Arabia \u2014 were willing to normalize ties before the conflict was resolved.\n\nYet Israelis \u2014 even those who oppose the occupation \u2014 by and large reject any contextualization of the Hamas attack as their military sets out to destroy the Islamic militant group.\n\nA month into the war, Israel is bracing for a long haul. Former Defense Minister Benny Gantz, now part of a special war Cabinet, has predicted the fighting could last a year or more.\n\nAs the military moves deeper into Gaza City, the epicenter of the Hamas command, casualties on both sides are expected to surge as combat moves into a dense urban landscape, with a warren of underground tunnels stocked with fighters and munitions.\n\nSo far, at least 30 Israeli soldiers have been killed since the ground operation began. Israel historically has had a low tolerance for casualties. Complicating matters are the hostage situation and the danger that the fighting will spiral into a devastating multifront conflict. Confrontations with militants in Lebanon, the West Bank, Syria and Yemen are already taking place.\n\n\u201cThe big question is, has Israeli society steeled itself on the question of casualties?\u201d Amos Harel, military correspondent for the Haaretz daily, told Army Radio. \u201cAfter the blow we took on Oct. 7, they may be willing. But after the news starts trickling in, and we understand that this is an invasion with bloodshed on both sides, it won\u2019t be easy to swallow at all.\u201d\n\n___\n\nFull AP coverage at https://apnews.com/hub/israel-hamas-war",
    "authors": [
        "Abc News"
    ],
    "publish_date": null,
    "top_image": "https://i.abcnewsfe.com/a/92d36590-d421-443e-9920-80ad9909256a/wirestory_0cebcbcf0550ee08c0d757334f69851d_16x9.jpg?w=992",
    "movies": [],
    "keywords": [],
    "summary": ""
}
"""
token_count = len(encoding.encode(text))
print(f"The text contains {token_count} tokens.")

