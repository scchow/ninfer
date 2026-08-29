#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <span>
#include <string>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;

const fi::ToolArgumentTypeContracts kNoTypeContracts;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

fi::ToolArgumentTypeContracts contracts_for(const std::string& tool_name, Json properties) {
    const std::string definition =
        Json{{"type", "function"},
             {"function", Json{{"name", tool_name},
                               {"parameters",
                                Json{{"type", "object"}, {"properties", std::move(properties)}}}}}}
            .dump();
    return fi::build_tool_call_output_contract(std::span<const std::string>(&definition, 1), true)
        ->argument_types;
}

int test_single_call() {
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("Calling weather.\n"
                                        "<tool_call>\n"
                                        "<function=get_weather>\n"
                                        "<parameter=city>\nParis\n</parameter>\n"
                                        "<parameter=days>\n2\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const fi::ParsedToolCallOutput parsed = fi::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_keeps_calls() {
    // Graceful degradation: a well-formed call followed by non-whitespace suffix keeps the
    // parsed call and returns the suffix as content (previously fell back to raw text).
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "well-formed call survives suffix");
    failures += check(parsed.tool_calls.size() == 1, "call kept despite suffix");
    failures += check(parsed.content == "extra answer", "suffix returned as content");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const fi::ParsedToolCallOutput anthropic =
        fi::parse_qwen_tool_call_output(text, 128, kNoTypeContracts);
    const fi::ParsedToolCallOutput openai =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const fi::ParsedToolCallOutput too_long =
        fi::parse_qwen_tool_call_output(too_long_text, 128, kNoTypeContracts);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_declared_strings_are_not_json_sniffed() {
    const auto contracts = contracts_for(
        "TaskUpdate",
        Json{{"taskId", Json{{"type", "string"}}},
             {"content", Json{{"type", "string"}}},
             {"truthy", Json{{"type", "string"}}},
             {"nullish", Json{{"type", "string"}}},
             {"quoted", Json{{"type", "string"}}},
             {"windows", Json{{"type", "string"}}},
             {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared-string tool call was not parsed");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("taskId").is_string() && args.at("taskId") == "1",
                      "numeric-shaped task ID was not preserved as a string");
    failures += check(args.at("content") == "  {\"x\":1}\n",
                      "string content lost meaningful whitespace or was JSON-decoded");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped strings were promoted");
    failures += check(args.at("quoted") == "\"literal\"",
                      "string payload was reinterpreted as embedded JSON");
    failures += check(args.at("windows") == "  value  ",
                      "CRLF framing or string spaces were not preserved");
    failures += check(args.at("string_or_number") == "7",
                      "string-admitting union destructively promoted raw text");
    return failures;
}

int test_declared_non_string_values_are_json_decoded() {
    const auto contracts = contracts_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}},
                          {"flag_or_null", Json{{"type", Json::array({"null", "boolean"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=count>\n7\n</parameter>\n"
                                        "<parameter=total>\n8\n</parameter>\n"
                                        "<parameter=ratio>\n1.5\n</parameter>\n"
                                        "<parameter=enabled>\ntrue\n</parameter>\n"
                                        "<parameter=payload>\n{\"x\":1}\n</parameter>\n"
                                        "<parameter=items>\n[\"a\",2]\n</parameter>\n"
                                        "<parameter=optional>\nnull\n</parameter>\n"
                                        "<parameter=flag_or_null>\nfalse\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("count").is_number_integer() && args.at("count") == 7,
                      "integer parameter was not decoded");
    failures += check(args.at("total").is_number_integer() && args.at("total") == 8,
                      "integer JSON value did not satisfy number schema");
    failures += check(args.at("ratio").is_number_float() && args.at("ratio") == 1.5,
                      "number parameter was not decoded");
    failures += check(args.at("enabled").is_boolean() && args.at("enabled") == true,
                      "boolean parameter was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object parameter was not decoded");
    failures += check(args.at("items").is_array() && args.at("items").at(1) == 2,
                      "array parameter was not decoded");
    failures += check(args.at("optional").is_null(), "declared nullable integer rejected null");
    failures += check(args.at("flag_or_null").is_boolean() && args.at("flag_or_null") == false,
                      "type-array order changed boolean interpretation");
    return failures;
}

int test_declared_type_mismatches_are_forwarded_without_coercion() {
    const auto contracts =
        contracts_for("configure", Json{{"object_as_integer", Json{{"type", "integer"}}},
                                        {"one_as_boolean", Json{{"type", "boolean"}}},
                                        {"string_as_boolean", Json{{"type", "boolean"}}},
                                        {"python_boolean", Json{{"type", "boolean"}}},
                                        {"null_as_boolean", Json{{"type", "boolean"}}}});

    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=object_as_integer>\n{}\n</parameter>\n"
                                        "<parameter=one_as_boolean>\n1\n</parameter>\n"
                                        "<parameter=string_as_boolean>\n\"true\"\n</parameter>\n"
                                        "<parameter=null_as_boolean>\nnull\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid JSON was rejected because it did not match the declared type");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("object_as_integer").is_object(),
                      "object-shaped JSON was coerced to the declared integer type");
    failures += check(args.at("one_as_boolean").is_number_integer(),
                      "numeric JSON was coerced to the declared boolean type");
    failures += check(args.at("string_as_boolean").is_string(),
                      "string JSON was coerced to the declared boolean type");
    failures += check(args.at("null_as_boolean").is_null(),
                      "null JSON was coerced to the declared boolean type");

    const std::string invalid =
        "<tool_call>\n<function=configure>\n<parameter=python_boolean>\nTrue\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto rejected = fi::parse_qwen_tool_call_output(invalid, 64, contracts);
    failures += check(!rejected.is_tool_call_response && rejected.content == invalid &&
                          rejected.tool_calls.empty(),
                      "non-JSON value for a declared non-string parameter did not fall back");
    return failures;
}

int test_unknown_schema_keeps_legacy_inference() {
    const auto contracts = contracts_for(
        "legacy", Json{{"missing_type", Json::object()}, {"invalid_type", Json{{"type", "int"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=legacy>\n"
                                        "<parameter=missing_type>\n7\n</parameter>\n"
                                        "<parameter=invalid_type>\n8\n</parameter>\n"
                                        "<parameter=undeclared>\n9\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("invalid_type") == 8 &&
                          args.at("undeclared") == 9,
                      "unknown-schema parameter changed legacy inference");
    return failures;
}

int test_parser_enforces_active_tool_set() {
    const auto contracts = contracts_for("declared", Json{{"value", Json{{"type", "string"}}}});
    const auto parsed    = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function=other>\n<parameter=value>\n1\n</parameter>\n"
           "</function>\n</tool_call>",
        64, contracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                      "undeclared tool escaped the active tool-name set");
    failures += check(parsed.content.find("<function=other>") != std::string::npos,
                      "undeclared tool was not preserved as ordinary content");
    return failures;
}

int test_incremental_filter_valid_tool() {
    fi::ToolCallOutputDecoder filter(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    auto terminal = filter.finish();
    visible += terminal.content;
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "get_weather",
              "valid tool filter did not retain the structured call");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish().content;

    fi::ToolCallOutputDecoder normal(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    failures += check(partial_restored == partial_original,
                      "partial marker mismatch did not preserve raw bytes");
    return failures;
}


int test_trailing_prose_keeps_calls() {
    // Model narrates after its tool call ("response", analysis text). The well-formed call
    // must still be salvaged; trailing prose becomes content instead of poisoning the parse.
    const std::string text = "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\nls\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "The exit code is 2 because src does not exist.";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "trailing prose keeps tool response");
    failures += check(parsed.tool_calls.size() == 1, "trailing prose keeps one call");
    failures += check(parsed.tool_calls[0].name == "bash", "salvaged call name");
    failures += check(parsed.content.find("exit code is 2") != std::string::npos,
                      "trailing prose preserved as content");
    return failures;
}

int test_prose_between_blocks_salvages_first() {
    // First block good, then prose, then a second marker that is never closed: first call
    // survives, prose and fragment become content.
    const std::string text = "<tool_call>\n"
                             "<function=a>\n"
                             "<parameter=x>1</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "thinking out loud <tool_call>\n<function=b>";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "prose between blocks keeps response");
    failures += check(parsed.tool_calls.size() == 1, "only first block salvaged");
    failures += check(parsed.tool_calls[0].name == "a", "first block name");
    failures += check(parsed.content.find("thinking out loud") != std::string::npos &&
                          parsed.content.find("<tool_call>") != std::string::npos,
                      "prose and unclosed fragment kept as content");
    return failures;
}

int test_all_blocks_bad_still_falls_back() {
    // A single un-closable block must still produce the raw-text fallback (no silent drop).
    const std::string text = "prefix\n<tool_call>\n<function=broken>\n<parameter=x>1";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "all-bad blocks fall back to text");
    failures += check(parsed.content == text, "fallback preserves full text");
    return failures;
}

// Reasoning-channel salvage: with thinking on, the model can skip think-close and emit its
// tool call while still in the reasoning channel. GenerationService::run re-runs this parser
// on outcome.reasoning when the content channel produced no tool response; these tests pin the
// shapes that shape must handle (see docs/arch/gotchas/pitfalls.md, NInfer leak entry).
int test_reasoning_leak_salvage_shape() {
    // Realistic leak: thinking preamble, then a complete block at the reasoning tail.
    const std::string reasoning = "Let me check what is in the directory first.\n"
                                  "<tool_call>\n"
                                  "<function=bash>\n"
                                  "<parameter=command>\nls /tmp\n</parameter>\n"
                                  "</function>\n"
                                  "</tool_call>";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(reasoning, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "reasoning leak parses as tool response");
    failures += check(parsed.tool_calls.size() == 1, "leaked call salvaged");
    failures += check(parsed.tool_calls[0].name == "bash", "salvaged call name");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("command") == "ls /tmp", "salvaged call arguments");
    failures += check(parsed.content == "Let me check what is in the directory first.",
                      "real reasoning preamble kept, XML stripped");
    return failures;
}

int test_reasoning_leak_multiple_blocks() {
    // Two complete blocks at the reasoning tail: both must be salvaged.
    const std::string reasoning = "I need two commands.\n"
                                  "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
                                  "<tool_call>\n<function=b>\n<parameter=y>\ntwo\n</parameter>\n</function>\n</tool_call>";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(reasoning, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multi-block reasoning leak is tool response");
    failures += check(parsed.tool_calls.size() == 2, "both leaked calls salvaged");
    failures += check(parsed.content == "I need two commands.", "preamble kept for multi-block");
    return failures;
}

int test_reasoning_without_tool_xml_untouched() {
    // No marker in reasoning: parser must be a no-op so healthy turns are untouched.
    const std::string reasoning = "Plain thinking text with no tool markers.\n";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(reasoning, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "plain reasoning is not a tool response");
    failures += check(parsed.content == reasoning, "plain reasoning preserved verbatim");
    failures += check(parsed.tool_calls.empty(), "no calls from plain reasoning");
    return failures;
}

int test_reasoning_truncated_block_no_salvage() {
    // Generation cut mid-call in the reasoning channel: no complete block, so nothing is
    // salvaged and the raw text stays as content (turn remains a normal stop).
    const std::string reasoning = "Let me run it.\n<tool_call>\n<function=bash>\n<parameter=command>\nls";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(reasoning, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "truncated reasoning block not salvaged");
    failures += check(parsed.content == reasoning, "truncated reasoning preserved verbatim");
    return failures;
}

int test_reasoning_channel_filter_holdback() {
    // Streaming hygiene: the same ToolCallOutputDecoder guards the reasoning channel. While a
    // trailing region could still be a tool call it must not be emitted; finish() either
    // salvages it into structured calls (held content dropped) or restores it verbatim.
    const std::string stream = "thinking out loud\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
                               "</parameter>\n</function>\n</tool_call>";
    fi::ToolCallOutputDecoder salvaged(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    for (std::size_t i = 0; i < stream.size(); i += 7) {
        visible += salvaged.feed(stream.substr(i, 7));
    }
    auto terminal = salvaged.finish();
    int failures = 0;
    failures += check(visible == "thinking out loud",
                      "reasoning decoder held back trailing tool region");
    failures += check(terminal.content.empty(), "salvaged reasoning region dropped at finish");
    failures +=
        check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "bash",
              "salvaged reasoning region produced a structured call");

    // A trailing region that is not a complete well-formed tool call (generation cut
    // mid-call) is restored verbatim so the turn stays a normal stop.
    const std::string truncated =
        "thinking out loud\n<tool_call>\n<function=bash>\n<parameter=command>\nls";
    fi::ToolCallOutputDecoder not_salvaged(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    for (std::size_t i = 0; i < truncated.size(); i += 7) {
        restored += not_salvaged.feed(truncated.substr(i, 7));
    }
    restored += not_salvaged.finish().content;
    failures += check(restored == truncated,
                      "non-salvaged reasoning region flushed verbatim at finish");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_keeps_calls();
    failures += test_configured_name_limit();
    failures += test_declared_strings_are_not_json_sniffed();
    failures += test_declared_non_string_values_are_json_decoded();
    failures += test_declared_type_mismatches_are_forwarded_without_coercion();
    failures += test_unknown_schema_keeps_legacy_inference();
    failures += test_parser_enforces_active_tool_set();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_trailing_prose_keeps_calls();
    failures += test_prose_between_blocks_salvages_first();
    failures += test_all_blocks_bad_still_falls_back();
    failures += test_reasoning_leak_salvage_shape();
    failures += test_reasoning_leak_multiple_blocks();
    failures += test_reasoning_without_tool_xml_untouched();
    failures += test_reasoning_truncated_block_no_salvage();
    failures += test_reasoning_channel_filter_holdback();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
