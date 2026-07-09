const { buildModule } = require("@nomicfoundation/hardhat-ignition/modules");

module.exports = buildModule("DragScoresModule", (m) => {
  // Hardhat account #11 == PRIVATE_KEYS.llm_service in attack/ssm_score and
  // the llm_service private key in drag_llm_service/configs/config.yaml.
  // Only this address may call feedbackAndUpdateScoreRecords (see
  // DragScores.llmService).
  const llmService = m.getAccount(11);
  const dragScores = m.contract("DragScores", [llmService]);
  return { dragScores };
});


