# Why these tools do not help with our analysis

The table in `image.png` comes from a chatbot answer about getting house dimensions from an address. Its claims are unverified. This note compares it with what LocationLabel has to measure.

## What we need to measure

| Measurement | Size of the thing | Precision needed |
|---|---|---|
| Curb ramp / driveway apron slope | a strip about 1.5 m wide that rises about 12 cm | a few degrees; a slope is an angle, so it needs the shape of the surface, not distances |
| Curb height | about 10 to 20 cm | centimetres |
| Stop sign height | about 2 to 3 m | a few centimetres |
| Pole height | a few metres | a few centimetres |

All of these are small, street-level, ground features. None of them is a building.

## Why each tool does not fit

| Tool | What it gives | Why it does not help with our analysis | A small possible use | Cost and access |
|---|---|---|---|---|
| **Nearmap** | Aerial and oblique imagery; building height, footprint, roof, walls, windows, doors | Looks from above, so a 12 cm rise on a 1.5 m wide ramp is not visible. It cannot see a stop sign face, a sign height or a curb face. Its measurements are built for roofs and walls. | Driveway width, length and area from above | Paid subscription |
| **EagleView One** | 3D exterior, roof, wall, window and door dimensions | Same limits as Nearmap: aerial, and made for buildings. No ground slope at centimetre level, nothing about signs, poles or curbs. | Roof and hardscape area, if that were ever wanted | Paid, sold to roofing and insurance users |
| **HOVER** | Detailed exterior dimensions and a 3D model of a house from phone photos | Needs a person at the house taking a guided set of photos, so it cannot run on addresses or on Street View images. It models building exteriors, not curb ramps or sign heights. | None as a tool. It shows that measuring from several photos works, which is the idea behind our `ramp` folder | Paid per property |
| **Google Street View "Insights"** | Street View images and camera information to run your own computer vision on | We already use Street View images and their camera data (position, heading). It adds no depth, so it does not remove the single-photo scale problem. I am not sure exactly what this product is. | None beyond what we already do | Google Maps Platform key and billing; you have no key |
| **Google Solar API** | Building and roof geometry; elevation data near buildings | Aimed at roofs. Its ground elevation is far too coarse to measure a ramp: a ramp is 1.5 m wide and rises 12 cm. Needs a Google key. | A rough overall driveway grade, as a plausibility check on our result | Google Maps Platform key and billing |
| **Street View Scale Measure** | A manual way to calibrate a Street View image against a known object | Manual, one image at a time, so it cannot be automated for many addresses. A single image still gives no reliable slope. | None | Free, manual |

## What is useful from the list

| Idea | Why it is useful | Needs |
|---|---|---|
| Public elevation data (for example USGS in the US) or the Solar API ground elevation | A rough driveway grade to cross-check our multi-photo slope | Only worth trying if our numbers disagree with a level measurement |
| Garage doors as a ruler | Standard doors are about 8 to 9 ft (one car) or 16 ft (two cars) wide and 7 ft tall, which could give a metric scale for driveway width or for single-photo estimates | Not needed for slope, which is an angle |

## Conclusion

None of these tools measures curb ramp slope, curb height, or sign and pole height. The best way forward is still several photos of the same ramp taken from different positions, checked against a few level measurements on site.

---

# Austin limits we could check our measurements against

Goal: when a measured value goes over a regional limit, show both the limit and the computed value.

Each row says where the number comes from and whether I read it myself.

**Reliability key**
- **Read:** I read the document itself.
- **Snippet:** it came from a search-result summary of an official page; I could not open the page.
- **Third party:** it comes from a website that summarizes the code; treat it as a lead to verify.
- **Not Austin:** a federal or state rule that applies in Austin but is not an Austin ordinance.

`library.municode.com` loads its text with JavaScript, so none of my tools could read it. Every Municode section below still needs checking by a person on the site.

## What is useful to us

| Limit | Value | Source | Can we measure it? |
|---|---|---|---|
| Driveway slope from the curb to the sidewalk zone | **10 % max** | Standard 433S-1A (read) | Yes. This is our multi-photo slope, and the most useful one. |
| Driveway slope in the sidewalk crossing strip (the "cross slope" the people walking along the sidewalk feel) | **2 % max**, over a strip at least 5 ft (1.5 m) deep | Standard 433S-1A (read) | Possible with the multi-photo method, but the strip is only 1.5 m deep and sits right behind the apron, so we would need the slope measured separately for each stretch. Not done yet. |
| Driveway slope past the property line | G1 plus D should not exceed **15 %** | Standard 433S-1A note 5 (read) | Partly. |
| Single-family driveway width | **10 to 20 ft** | Standard 433S-1A (read) | Not yet. It needs a metric scale, which the multi-view method can give from camera height. |
| Curb ramp running and cross slope | 8.3 % and 2 % | ADA / Texas rules, not an Austin ordinance | Yes, already in our config. |
| Stop sign height | 7 ft (2.13 m) minimum | Texas MUTCD, not Austin | Yes, already in our config. |

## All the limits found

| What | Limit | Where it applies | Source | Reliability |
|---|---|---|---|---|
| Driveway slope between the curb and the sidewalk zone | **10 % maximum** | the sloped part of a driveway approach (apron) | City of Austin Standard **433S-1A**, "Flared Type I Driveway", adopted 4/14/2025, sheet 3 | Read |
| Driveway slope in the sidewalk crossing strip | **2 % maximum**, over a strip at least **5 ft** deep (the pedestrian crossing); it is the slope toward the street, which is the sidewalk's own cross slope | where the driveway crosses the sidewalk | Standard 433S-1A, sheets 1 and 3 | Read |
| Driveway slope beyond the right-of-way line | "G1 plus D should not exceed **15 %**" | private-property side; owner is responsible for grade breaks. If G2 is over 15 %, consult the Fire Department | Standard 433S-1A, note 5 | Read |
| Single-family driveway width | **10 ft minimum, 20 ft maximum** | at the driveway | Standard 433S-1A, sheet 1 table | Read |
| Driveway width at the property line | 10 to 20 ft | Austin right-of-way residential review guidelines | search summary | Snippet |
| Sidewalk through a driveway | 5 ft minimum width; see TCM Section 4 | | Standard 433S-1A, sheet 1 and note 13 | Read |
| Driveway height above the gutter | typically at least **150 mm (6 in) plus 20 mm per metre (1/4 in per foot)** over the distance from the curb to the right-of-way line; all driveways slope toward the street | | Standard 433S-1A, sheet 1 note | Read |
| Curb height at a driveway | tapers to **zero** at the end of the flare ("zero curb") | | Standard 433S-1A, note 1 | Read |
| Driveway grade at a sidewalk (older wording) | 2 % maximum over a 3 ft throat; G1 not over 10 %, and over 10 % needs a 50 ft tangent | | Transportation Criteria Manual 5.3.1 | Snippet (mirror site) |
| Curb ramp running slope | **8.3 % (1:12) maximum** | curb ramps | ADA / TAS; Texas DOT manual | Not Austin |
| Curb ramp cross slope | **2 % maximum** at stop or yield controlled crossings (the same source gives 5 % at signalized or uncontrolled ones) | curb ramps | Texas DOT manual | Not Austin |
| Sign height | **7 ft (2.13 m) minimum** above the ground, and 7 to 7.5 ft for roadside signs | stop signs | Texas MUTCD / TxDOT sign guidance | Not Austin |
| Maximum driveway grade in certain watersheds | 14 % (25 ft exception) | environmental watershed rules | Code 25-6-441, 25-6-442, cited by a summary site | Third party |
| Sidewalks and driveways height above adjacent grade | 30 in maximum | | Austin code, from a summary | Snippet |

## What I found no limit for

| What | Result |
|---|---|
| **Pole height** | No Austin limit found for utility poles, street light poles or sign posts. Poles are not normally limited by height in the city code. Sign mounting height is covered above. |
| **Curb height** | No Austin limit found. The standard drawing only says it tapers to zero at driveways. |

## What is not useful to us

Concrete thickness, reinforcement, sand bedding, joints, spacing between driveways, parking and fence rules. They describe how things are built or laid out, not anything a street-level photo shows.

## Other places for state and city provisions

| Source | Why it could matter |
|---|---|
| Texas Accessibility Standards (TAS), Texas Department of Licensing and Regulation | The state version of the ADA; ramp and sidewalk slopes |
| Texas MUTCD | Sign mounting heights and placement |
| Austin Transportation Criteria Manual, Sections 4 and 7 | Sidewalk width; other driveway rules (the standard points here) |
| Austin Land Development Code, Chapter 25-6 | Driveway permits and rules |

## How "show the limit and the computed value" could work

1. **A region file for Austin:** each rule has what it applies to, the limit, the document and section, and whether it is a maximum or a minimum. Another region later means another file.
2. **When a value exceeds a limit,** the label shows both, for example the computed 6.8 degrees (12 %) with "Austin limit 5.7 degrees (10 %)" under it.
3. **It has to stay honest about error.** Our slopes are about 1 to 2 % off in the best test, so near a limit the label should say "may exceed", not "exceeds". The existing "beyond the uncertainty" check can do this.
4. **The 10 % limit covers one segment,** from the curb to the sidewalk zone, not the whole driveway. If our measured patch covers more than that, the comparison can be wrong, so the label should say which patch it measured.
5. **These labels are not compliance findings.** They are estimates from photos, and the app should say so.

## What "10 %" means

**10 % is a slope: the surface rises 10 units for every 100 units you move horizontally.** It is a ratio, not an angle in degrees.

Examples:
- 1 m along the ground: at most 10 cm of rise.
- A 1.5 m long driveway apron: at most 15 cm (about 6 in) of rise, from the gutter up to the sidewalk.
- Written as a ratio: 1:10. For comparison, the ADA ramp limit of 8.33 % is 1:12.

The same numbers as angles (our labels show degrees):

| Slope | Ratio | Angle |
|---|---|---|
| 2 % | 1:50 | 1.1 degrees |
| 5 % | 1:20 | 2.9 degrees |
| 8.33 % (ADA ramp limit) | 1:12 | 4.8 degrees |
| **10 % (Austin driveway apron limit)** | 1:10 | **5.7 degrees** |
| 12 % | 1:8.3 | 6.8 degrees |
| 15 % (Austin limit past the property line) | 1:6.7 | 8.5 degrees |

So when a label shows a driveway at 6.8 degrees (12 %), it is above the 5.7 degree (10 %) limit.

**What the 10 % covers:** the sloped part of the driveway between the curb and the sidewalk zone (Austin Standard 433S-1A). It does not cover the whole driveway. The flat strip where the driveway crosses the sidewalk has a stricter limit of 2 %.

**To convert:** angle = arctan(percent / 100), and percent = tan(angle) x 100.

## The three driveway slopes explained, with pictures

**Correction.** An earlier explanation of mine said the 2 % "cross slope" is measured across the width of the driveway. That was wrong. In Austin's drawing, the 2 % is the slope **toward the street** inside the strip where the driveway crosses the sidewalk. It is called a cross slope because, for a person walking along the sidewalk, tilting toward the street is a sideways tilt. It is the same direction as the apron slope, just on a different stretch of the driveway.

![Side view of a driveway with the three slopes](images/1_driveway_side_view.png)

![Plan view: where the 10 % and 2 % limits apply](images/2_driveway_plan_view.png)

| | Where | Direction | Limit | Why |
|---|---|---|---|---|
| **1. Slope from the curb** (the apron) | from the gutter up to the sidewalk | toward the street | 10 % | A steeper apron scrapes cars and is hard for wheelchairs and walkers |
| **2. Sidewalk crossing strip** | a strip at least 5 ft deep where the driveway crosses the sidewalk | toward the street (the sidewalk's cross slope) | 2 % | Someone rolling along the sidewalk is pushed toward the street by any tilt |
| **3. Past the property line** | on the private part of the driveway | along the driveway | G1 + D at most 15 % | A sharp change in grade scrapes the underside of a car |

![Why the crossing must be almost level](images/3_why_the_crossing_is_nearly_level.png)

**Item 3 in plain words.** Past the property line the driveway has a first grade (**G1**), then it changes to a second, usually steeper grade (**G2**). **D** is how much the grade changes at that break. The standard says G1 plus D should not exceed 15 %, which means the steeper grade after the break should not be above 15 %. If it is, the standard says to consult the Fire Department, since fire trucks need to get in.

![A car crossing a gentle and a sharp grade break](images/4_grade_break_scrape.png)

**What this means for our measurements**
- **Item 1 (the apron, 10 %):** this is the slope our multi-photo method measures best.
- **Item 2 (the crossing strip, 2 %):** it needs the slope measured separately for the strip behind the apron. We do not do that yet.
- **Item 3 (past the property line):** mostly out of sight from the street, so we can usually not check it.
- **Our "cross slope" number** (slope along the curb, side to side across the driveway) is not one of Austin's driveway limits.

The pictures are redrawn by hand from Standard 433S-1A, with the heights exaggerated, and are not to scale.
