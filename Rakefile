# filename: Rakefile
require "html-proofer" # Require gem for using within tasks

task :default do
  puts "Running CI tasks..."
  touch ./_site/.nojekyll
  sh("JEKYLL_ENV=production bundle exec jekyll build")
  # Add HTMLProofer.check_directory("./_site").run in order to start checking
  # for invalid HTML
  HTMLProofer.check_directory(
    "./_site",
    url_ignore: [/linkedin.com|php-fig.org|bower.io|bost.ocks.org|elementary.io/]
  ).run
  puts "Jekyll successfully built"
end
