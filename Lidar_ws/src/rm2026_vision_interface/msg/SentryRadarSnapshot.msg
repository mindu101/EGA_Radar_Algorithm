# One committed radar frame for sentry_decide. Never mix fields across frames.
uint8 FACTION_UNKNOWN=0
uint8 FACTION_RED=1
uint8 FACTION_BLUE=2
std_msgs/Header header
uint32 pipeline_epoch
uint64 sequence
uint8 faction
# Age of the input frame at publication, measured by the producer's clock.
float32 source_age_ms
SentryRadarEnemy[5] enemies
SentryRadarMate[5] mates
SentryRadarOutpost[2] outposts
# -1 means no reliable kill count is available.
int32 kills
