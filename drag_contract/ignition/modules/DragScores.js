const { buildModule } = require("@nomicfoundation/hardhat-ignition/modules");

module.exports = buildModule("DragScoresModule", (m) => {
  const dragScores = m.contract("DragScores", []);
  return { dragScores };
});


