/** Request / response shapes. They follow TypeSafe Jev's `system_one` API, which Ped reproduces. */

export type QuestionType = "choice" | "score" | "noul";

export interface ChoiceQuestion {
  type: "choice";
  instructions: string | object;
  /** option -> short description (or null), or a plain list of option names */
  criteria: Record<string, string | null> | readonly string[];
}

export interface ScoreQuestion {
  type: "score";
  instructions: string | object;
  /** ordered levels, index 0 = lowest */
  criteria: readonly string[];
}

export interface NoulQuestion {
  type: "noul";
  instructions: string | object;
  criteria?: { true?: string; false?: string };
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  /** 1 - normalized entropy of the answer distribution: how concentrated it is (not calibrated) */
  confidence: number;
  /** probability of the reported answer, max(p): the number to gate on once temperatures are calibrated */
  answer_confidence: number;
  rl_agent: { act_probability: number };
  /** set by `systemOneLong` on a document read in several windows: the window that decided */
  window?: WindowInfo;
}

export interface ScoreAnswer {
  type: "score";
  /** expected level (0 .. levels-1) */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  /** 1 - normalized entropy of the level distribution */
  confidence: number;
  /** probability of the most likely level */
  answer_confidence: number;
  rl_agent: { act_probability: number };
  /** set by `systemOneLong` on a document read in several windows: the window that decided */
  window?: WindowInfo;
}

export interface NoulAnswer {
  type: "noul";
  /** P(true) */
  noul: number;
  /** max(P(true), P(false)) */
  confidence: number;
  /** probability of the reported answer, max(P(true), P(false)), as on the other types */
  answer_confidence: number;
  rl_agent: { act_probability: number };
  /** set by `systemOneLong` on a document read in several windows: the window that decided */
  window?: WindowInfo;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion ? ChoiceAnswer : Q extends ScoreQuestion ? ScoreAnswer : NoulAnswer;

/** How much of the state a sequence kept (the upstream truncation report, issue #174). */
export interface StateStats {
  state_tokens: number;
  state_tokens_used: number;
  state_tokens_dropped: number;
  truncated: boolean;
}

/** What the head budget did to a question's options: `options_distinct` < `options` means two options collapsed. */
export interface OptionStats {
  options: number;
  options_distinct: number;
  tokens_per_option: number | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  /** tokens of the serialized state */
  state_tokens: number;
  /** state tokens cut to fit the sequence, for the question that lost the most */
  state_tokens_dropped: number;
  /** the state did not fit: the model did not read all of it (see `systemOneLong`) */
  truncated: boolean;
  truncated_questions: string[];
  /** only when a question's options lost their own token span to the head budget */
  options?: Record<string, OptionStats>;
}

export interface SystemOneResult<Q extends Record<string, Question>> {
  model: string;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: Usage;
}

/** Which window of a long document decided an answer: token offsets into the tokenized state. */
export interface WindowInfo {
  index: number;
  token_start: number;
  token_end: number;
  count: number;
}

export interface LongUsage extends Omit<Usage, "truncated"> {
  /** number of windows that were themselves truncated */
  truncated: number;
  /** windows the document was read in (1 when it fits one window) */
  windows: number;
}

export interface SystemOneLongResult<Q extends Record<string, Question>> {
  model: string;
  /** the deciding window's answer; `window` is absent when the document fit one window */
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: LongUsage;
}

/** ped_config.json, written by export/export_onnx.py from the checkpoint's rl_agent_config.json */
export interface PedConfig {
  max_len: number;
  head_max_len: number;
  temperature: [number, number, number];
  temperature_by_options: Record<string, number>;
}
