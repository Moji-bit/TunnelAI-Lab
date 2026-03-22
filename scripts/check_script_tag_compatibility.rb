#!/usr/bin/env ruby
# frozen_string_literal: true

require 'yaml'
require 'set'

ROOT = File.expand_path('..', __dir__)
TAGS_FILE = File.join(ROOT, 'tags', 'tags.yaml')
SCRIPTS_DIR = File.join(ROOT, 'scripts')

LEGACY_TAGS = Set[
  'Z2.TRAF.Speed',
  'Z2.TRAF.Density',
  'Z2.CO.S01.Value',
  'Z2.VIS.S01.Value',
  'Z2.VMS.SpeedLimit',
  'Z2.FAN.StageCmd',
  'Z2.EVENT.IncidentFlag'
].freeze

unless File.exist?(TAGS_FILE)
  warn "ERROR: missing tags file #{TAGS_FILE}"
  exit 2
end

yaml_tags = YAML.load_file(TAGS_FILE).fetch('tags', []).map { |t| t['tag_id'] }.to_set

errors = []
legacy_hits = []
summary = []

Dir[File.join(SCRIPTS_DIR, '*.py')].sort.each do |path|
  content = File.read(path)
  refs = content.scan(/['\"](Z[1-4]\.[^'\"\s]+)['\"]/).flatten.uniq.sort
  next if refs.empty?

  missing = refs.reject { |r| yaml_tags.include?(r) || LEGACY_TAGS.include?(r) }
  legacy = refs.select { |r| LEGACY_TAGS.include?(r) }

  summary << [File.basename(path), refs.size, legacy.size, missing.size]
  legacy_hits << [File.basename(path), legacy] unless legacy.empty?
  errors << [File.basename(path), missing] unless missing.empty?
end

puts "Tag source: #{TAGS_FILE}"
puts "Python scripts: #{SCRIPTS_DIR}"
puts
puts 'Per-file summary (refs / legacy / unknown):'
summary.each do |fname, refs, legacy, missing|
  puts format('  - %-30s refs=%-3d legacy=%-2d unknown=%-2d', fname, refs, legacy, missing)
end

unless legacy_hits.empty?
  puts "\nLegacy tag references (allowed but should be migrated):"
  legacy_hits.each do |fname, tags|
    puts "  - #{fname}"
    tags.each { |t| puts "      * #{t}" }
  end
end

if errors.empty?
  puts "\nCompatibility result: PASS (no unknown tag references in scripts/*.py)"
  exit 0
end

puts "\nUnknown tag references (not in tags.yaml and not in legacy allowlist):"
errors.each do |fname, tags|
  puts "  - #{fname}"
  tags.each { |t| puts "      * #{t}" }
end
puts "\nCompatibility result: FAIL"
exit 1
