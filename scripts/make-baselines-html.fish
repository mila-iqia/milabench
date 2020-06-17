#!/usr/bin/env fish

for f in (ls baselines/**.json)
    set h (string replace -r "\.json" "" -- $f)
    set title (string replace -r "[^/]*/" "" -- $f)
    set b (string replace -r ".*/" "" -- (dirname $f))
    milarun report $f --html $h.html --weights weights/$b.json --title $title
end
