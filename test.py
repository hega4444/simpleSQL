import tiktoken

import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = """
<article><content>[1/2] General view of Aramco's Ras Tanura oil refinery and oil terminal in Saudi Arabia May 21, 2018. Picture taken May 21, 2018. REUTERS/Ahmed Jadallah/ File Photo Acquire Licensing Rights
WASHINGTON, Oct 31 (Reuters) - (This Oct. 30 story has been corrected to change the year to 2024 in paragraph 1)
The World Bank said on Monday it expected global oil prices to average $90 a barrel in the fourth quarter and fall to an average of $81 in 2024 as slowing growth eases demand, but warned that an escalation of the latest Middle East conflict could spike prices significantly higher.
The World Bank's latest Commodity Markets Outlook report noted that oil prices have risen only about 6% since the start of the Israel-Hamas war, while prices of agricultural commodities, most metals and other commodities "have barely budged."
The report outlines three risk scenarios based on historical episodes involving regional conflicts since the 1970s, with increasing severity and consequences.
A "small disruption" scenario equivalent to the reduction in oil output seen during the Libyan civil war in 2011 of about 500,000 to 2 million barrels per day (bpd) would drive oil prices up to a range of $93 to $102 a barrel in the fourth quarter, the bank said.
A "medium disruption" scenario - roughly equivalent to the Iraq war in 2003 - would cut global oil supplies by 3 million to 5 million bpd, pushing prices to between $109 and $121 per barrel.
The World Bank's "large disruption" scenario approximates the impact of the 1973 Arab oil embargo, shrinking the global oil supply by 6 million to 8 million bpd. This would initially drive up prices to $140 to $157 a barrel, a jump of up to 75%.
"Higher oil prices, if sustained, inevitably mean higher food prices," said Ayhan Kose, the World Bank’s Deputy Chief Economist. "If a severe oil-price shock materializes, it would push up food price inflation that has already been elevated in many developing countries."
The World Bank report said that China's oil demand was surprisingly resilient given strains in the country's real estate sector, rising 12% in the first nine months of 2023 over the same period of 2022.
Oil production and exports from Russia have been relatively stable this year despite Western-imposed embargoes on Russian crude to punish Moscow over its invasion of Ukraine, the World Bank said.
RUSSIAN PRICE CAP 'UNENFORCEABLE'
Russia's exports to the European Union, the U.S., Britain and other Western countries fell by 53 percentage points between 2021 and 2023, but these have been largely replaced with increased exports to China, India and Turkey - up 40 percentage points over the same period.
"The price cap on Russian crude oil introduced in late 2022 appears increasingly unenforceable given the recent spike in Urals prices," the World Bank said, referring to the benchmark Russian crude, currently quoted in the mid-$70s per barrel range, well above the G7-led $60 price cap for Russian crude. The cap aims to deny buyers of Russian crude the use of Western-supplied services, including shipping and insurance, unless cargoes are sold at or below the capped price.
"It seems that by putting together a "shadow fleet" (of tankers), Russia has been able to trade outside of the cap; the official Urals benchmark recently breached the cap for more than three months, averaging $80 per barrel in August," the report said.
If the Israel-Hamas conflict escalates, policymakers in developing countries will need to take steps to manage a potential increase in headline inflation, the World Bank said. It added that governments should avoid trade restrictions such as export bans on food and fertilizer because they can often intensify price volatility and heighten food insecurity.
Reporting by David Lawder; Editing by Christian Schmollinger and Louise Heavens
Our Standards: The Thomson Reuters Trust Principles.</content></article>
"""
token_count = len(encoding.encode(text))
print(f"The text contains {token_count} tokens.")