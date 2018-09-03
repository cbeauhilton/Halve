#!/bin/bash

# Run this script in the root directory of your Jekyll source
# repository for your GitHub page, and it will build your site
# and push the resulting HTML to your remote master branch,
# i.e. to your site, with commit message given by the first
# argument

# So if I wanted to build my site locally and push it to my repository with the commit message “Latest build”, I would run: jekgit.sh "Latest build"

# from https://gist.github.com/drewsberry/1b9fc80682edd8bcecc4

if [[ -z "$1" ]]; then
  echo "Please enter a git commit message"
  exit
fi

jekyll build && \
  touch _site/.nojekyll && \
  cd _site && \
  git add . && \
  git commit -am "$1" && \
  git push origin master && \
  cd .. && \
  echo "Successfully built and pushed to GitHub."
