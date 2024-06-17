# Goal

Track any subjectively note-worthy observations resulting from the one-ways analysis, in alphabetical order.

Note-worthy is split into two categories:
1. Importance/clear signal
2. Data quality issues

Important to remember we use pandas quantile to determine bounds - unclear how this works for < 100 samples.

Interesting that there is no clear nice pattern like for the paper's trend and seasonality - maybe as we are aggregating weekly instead of monthly (as in paper).

Notice how these graphs are hard to read when super fine (1W) timescale (apposed to 1M).

Missingness is maybe highly correlated between variables (esp. modern and old).

Is meaning of missingness changing between modern and old? Especially seems different for YAD and main driver; difference between data gathered and YAD not existing

# Sampled 10% - Weekly aggregate

## Claim_Reparier_Approved
Note the typo in the column name
2. Clearly exhibits mid 2015 cutoff; likely older data was in different system. Call all times after this cutoff as 'modern' and before the cutoff as 'old'

1. Approved repairs and Non Approved are roughly same mean but Non approved much more variable
1. Unknown category still dwarfs everything - suspect this is just the missing data category. In this instance we can see stable and low mean relative to repairs that are filled in. I.e. presence of this data is likely biasing response up.

## Claim_Type
2. All claim types significantly less in 2020
2. W claim costs seem capped/thresholded from mid 2020 onwards; new policy?

1. F/T are both very rare but exhibit clearly higher overall average costs
1. W is significantly cheaper than everything else, 

## Claimant_DamageArea
2. Unknown fraction significantly drops in mid 2015; and average of modern unknowns is lower than others that are consistently present

1. Front damage is more expensive than rear damage - both in mean and 95%
1. Side is marginally cheaper than Front - much rarer
1. Roof, Locks, Mechanical, Underneath very rare - kinda pointless to keep; maybe mechanical is slightly more expensive

## Claimant_Type
2. Massive steep drop off in unknown near end of 2015
2. Is DRV|OWN different from OWN|DRV; former is much rarer and has higher av. payments
2. Again steep dropoff in some popular categories in 2020; e.g. OWN|DRV 

1. OWN marginally more expensive than OWN|DRV
1. Other categories seem noisy

## Claimant_VehicleMake
2. Huge drop in late 2015 of Unknown

1. Weirdly not a huge difference in BMW vs Ford or Vauxhall; I would guess BMW is slightly more expensive overall
1. Expensive cars exhibit more extreme values but otherwise trend quite similarily; no obvious inflation

## Claimant_VehicleModel
2. Huge drop in late 2015 of Unknown

1. Not clear if this is different from Claimant_VehicleMake

## CLLITF
IIRC this is related to legal disputes
2. So rare it seems no worth considering; even if more expensive in the few observed cases

## Complaint_Outcome
2. No drop modern vs old

1. If a complaint was made in general seems more expensive; irrespective if upheld or not

## Complaint
2. Highly correlated to Complaint_Outcome

## Country
2. Huge drop in late 2015 of Unknown

1. Not clear that any one country is more expensive

## Damage_Area
2. Identical to Claimant_DamageArea?

## Damage_Severity
2. Huge drop in late 2015 of Unknown; although still commonly missing

1. Clear ordinality in means; follows naive expectations

## Delivery_work
2. Huge drop in late 2015 of Unknown

1. No clear signal

## Driving_at_night
2. Huge drop in late 2015 of Unknown

1. No clear signal

## First_Liability
2. No drop off; gradually decreasing unknowns

1. Not clear what acronyms mean; but I looks meaningfully more than T; unsure otherwise

## Fraud_Outcome
2. Seems to essentially stop existing after late 2016

## full_occ
2. Huge drop in late 2015 of Unknown

1. No obvious job with higher trend; maybe company director? 

## Handling_Money
2. Huge drop in late 2015 of Unknown

1. Nothing obvious

## Involves_Sales
2. Huge drop in late 2015 of Unknown

1. Nothing obvious


# Sampled 10% - Monthly aggregate

## full_occ
Check this to validate findings from above are valid on coarser timescale
2. Huge drop in late 2015 of Unknown

1. No obvious job with higher trend; maybe company director? 
1. Missing noticably cheaper

## Involves_time_away
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper


## Job_involves_driving
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## Last_Liability
2. No drop

1. I is clearly most expensive; Unknown seems to be clearly cheapest

## Main_Commute
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

# Complete sample - Monthly aggregate

## Main_Lic_typ
2. Huge drop in late 2015 of Unknown

1. Provisional unsurpisingly more expensive; but very rare

## Main_Own_oth_veh
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## Main_Sex
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## MHCOVR
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. TPFT more expensive; probably as the claim is more likely to be severe

## MHCUSE
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Funky Class 2?

## MHNCDP
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## MHNCDY
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Weak but clear ordinal effect on response; as NCD years increases; average claim value decreases

## MHOTHV
Surprisingly common to have other vehicles
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## MHVMOD
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Clearly different behaviour between yes and no - not sure if it is obvious

## Motor Trade
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Again clearly different behaviour between yes and no but no a massive ultimate mean difference; likely related to freq

## NEW_Alt_Pcode_Ind
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Again clearly different behaviour between yes and no but no a massive ultimate mean difference; likely related to freq

## PCode
2. No missings

1. Some postcodes noticeably more cheap and stable; e.g. CM1 (Chelmsford) vs B33 (Birmingham)

## Policy_Cover
2. No missings

1. Ordinal relationship with 1/2/3, where 1 is cheapest - 4 and 6 so rare that may as well be irrelevant;

## Policy_Make
2. Seems like no missing

1. Unsurprising relationship with relations like Jaguar > Audi > Ford

## Policy_Model
So high cardinality unreasonable to draw any conclusions

## Policy_PhoneFlagCode
2. Huge drop in late 2015 of Unknown
2. Noticeable gradual drop in 'M'?

1. Modern missing noticably cheaper
1. Again clearly different behaviour between yes and no but no a massive ultimate mean difference; likely related to freq

## Policy_Status
2. More recent accidents with cancelled policy status more expensive than other categories - could be post accident cancellation?

## Possible_Fraud
2. No Y after late 2016

1. Hugely informative signal for increase in cost

## PCode_Repairer
2. Seems broadly absent before mid 2015

1. Modern claims no obvious correlation with location

## Product
2. Huge drop in late 2015 of Unknown
2. Clearly some policies are discontinued: e.g. T3_RN 

1. Modern missing noticably cheaper
1. T6_NB has higher extremal values

## Professional_Qualificiation_Require
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Maybe is only category that has something noticeably different - higher extremal values

## PropMain
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## propmar
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. Civil P and Widowed seem like minor risk factors - could be due to low sample size

## Rating_VehicleABI_TermCode
2. No missing
2. 9 is only old code present; weird spike in 9s mid 2015

1. B and G look good; I looks bad

## RD_Gender
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper
1. M slightly more expensive?

## Repair_or_loss
2. No missing

1. Huge signal - TL should be noticeably more

## Repair_TotalLoss
Not clear why this is so many categories or how it differs from above
2. C and D seem like discontinued categories; are N and S replacements?
2. T super rare and discontinued

1. U is a substantial increase over all others but v rare

## Repairer_type
1. Missing noticably cheaper; no drop or weird behaviour
1. Non approved might seem worse but also small sample size; means are slightly different

## Salvage_Category
2. C and D seem discontinued; repalced by N and S?
2. Is this just the same as Repair_TotalLoss?

## Unemp_Any
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## Unemp_Main
Interesingly no obvious bias for unemployed being more or less expensive
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## Unemp_YAD
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## Vehicle_PrivatePlate
2. Huge drop in late 2015 of Unknown

1. Modern missing noticably cheaper

## WD_Approved
1. Very confusing column but clearly high signal. Is unknown same as no? Or not relevant because no WD?

## WindscreenRepair
2. Clearly mix of WD claims and non WD claims makes this field a little messy; Repair weirdly more expensive than replace.

## YAD_Commute
2. Huge drop in late 2015 of Unknown

1. I would judge No as slightly more expensive but really no clear signal

## YAD_Delivery_work
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Driving_at_night
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Handling_Money
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Involves_Sales
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Involves_time_away
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Job_involves_driving
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_LIC_Typ
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean; maybe provisional more expensive (also unsurprisingly way more common to be provisional than for PH)

## YAD_Motor_Trade
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean; although Yes seems to have different behaviour from everything else

## YAD_occ_cat
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Own_oth_veh
Crazy that so many YAD own another vehicle
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Profession_Qualification_Req
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean; although Maybe seems to have different behaviour

## YAD_Sex
2. Huge drop in late 2015 of Unknown

1. No clearly most important in shifting mean

## YAD_Use
2. Huge drop in late 2015 of Unknown

1. N is clearly important if rare category

