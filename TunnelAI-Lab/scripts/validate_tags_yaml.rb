#!/usr/bin/env ruby
# frozen_string_literal: true

require 'yaml'
require 'set'

file = ARGV[0] || File.join(__dir__, '..', 'tags', 'tags.yaml')

unless File.exist?(file)
  warn "ERROR: file not found: #{file}"
  exit 2
end

data = YAML.load_file(file)
tags = data['tags'] || []
segments = data.dig('tunnel_profile', 'segments').to_i
fans = data.dig('tunnel_profile', 'fans').to_i
vms = data.dig('tunnel_profile', 'vms').to_i

errors = []
warnings = []

ids = tags.map { |t| t['tag_id'] }
dup_ids = ids.tally.select { |_k, v| v > 1 }
errors << "Duplicate tag_id values: #{dup_ids.keys.join(', ')}" unless dup_ids.empty?

pattern = /^Z([1-4])\.[A-Z]+\./

seen_segments = Set.new
seen_fans = Set.new
seen_vms = Set.new

tags.each do |tag|
  tag_id = tag['tag_id']
  zone = tag['zone']

  unless tag_id.is_a?(String) && zone.is_a?(Integer)
    errors << "Missing/invalid tag_id or zone in entry: #{tag.inspect}"
    next
  end

  m = tag_id.match(pattern)
  if m.nil?
    errors << "Invalid tag_id prefix format: #{tag_id}"
  elsif m[1].to_i != zone
    errors << "Zone mismatch: tag_id=#{tag_id} has zone #{m[1]}, field has #{zone}"
  end

  if (seg = tag_id[/\.S(\d{2})\./, 1])
    seg_i = seg.to_i
    seen_segments << seg_i
    errors << "Segment out of range in #{tag_id} (expected 1..#{segments})" unless (1..segments).include?(seg_i)
  end

  if (fan = tag_id[/\.F(\d{2})\./, 1])
    fan_i = fan.to_i
    seen_fans << fan_i
    errors << "Fan index out of range in #{tag_id} (expected 1..#{fans})" unless (1..fans).include?(fan_i)
  end

  if (vms_i = tag_id[/\.V(\d{2})\./, 1])
    vms_idx = vms_i.to_i
    seen_vms << vms_idx
    errors << "VMS index out of range in #{tag_id} (expected 1..#{vms})" unless (1..vms).include?(vms_idx)
  end

  if tag.key?('limits')
    min = tag.dig('limits', 'min')
    max = tag.dig('limits', 'max')
    if min.nil? || max.nil?
      errors << "limits must contain min and max: #{tag_id}"
    elsif min > max
      errors << "limits min > max in #{tag_id}: #{min} > #{max}"
    end
  end
end

warnings << "No segment tags discovered" if seen_segments.empty?
warnings << "No fan tags discovered" if seen_fans.empty?
warnings << "No VMS tags discovered" if seen_vms.empty?

puts "File: #{file}"
puts "Total tags: #{tags.size}"
puts "Unique tags: #{ids.uniq.size}"
puts "Covered segments: #{seen_segments.to_a.sort.join(', ')}"
puts "Covered fans: #{seen_fans.to_a.sort.join(', ')}"
puts "Covered VMS: #{seen_vms.to_a.sort.join(', ')}"

unless warnings.empty?
  puts "\nWarnings (#{warnings.size}):"
  warnings.each { |w| puts "  - #{w}" }
end

if errors.empty?
  puts "\nValidation result: PASS"
  exit 0
end

puts "\nErrors (#{errors.size}):"
errors.each { |e| puts "  - #{e}" }
puts "\nValidation result: FAIL"
exit 1
