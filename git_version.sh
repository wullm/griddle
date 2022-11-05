#!/bin/bash
cp -f include/git_version.h.format include/git_version.h

GIT_BRANCH=$(git branch 2>/dev/null | sed -n 's/^\* \(.*\)/\1/p' 2>/dev/null)
GIT_DATE=$(git log --pretty=format:%ad -n 1 2> /dev/null)
GIT_COMMIT=$(git describe --always --dirty="-dirty" 2> /dev/null)
GIT_MESSAGE=$(git log --pretty=format:%s -n 1 | cut -c 1-50)

# Needs escaping for multi-line output
GIT_STATUS=$(git status -bsuno)
IFS= read -d '' -r < <(sed -e ':a' -e '$!{N;ba' -e '}' -e 's/[&/\]/\\&/g; s/\n/ \\\\n/g' <<<"$GIT_STATUS")
GIT_STATUS=${REPLY%$'\n'}

sed -i "s/\"GIT_BRANCH\"/\"$GIT_BRANCH\"/g" include/git_version.h
sed -i "s/\"GIT_DATE\"/\"$GIT_DATE\"/g" include/git_version.h
sed -i "s/\"GIT_COMMIT\"/\"$GIT_COMMIT\"/g" include/git_version.h
sed -i "s/\"GIT_MESSAGE\"/\"$GIT_MESSAGE\"/g" include/git_version.h
sed -i "s/\"GIT_STATUS\"/\"$GIT_STATUS\"/g" include/git_version.h
