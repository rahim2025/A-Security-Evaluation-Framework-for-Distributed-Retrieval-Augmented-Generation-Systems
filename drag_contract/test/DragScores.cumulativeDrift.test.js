const { expect } = require("chai");
const { ethers, network } = require("hardhat");

// Tests the patient-throttling defense: feedbackAndUpdateScoreRecords bounds
// NET score movement over a rolling EPOCH_DURATION window, closing the gap
// where a per-tx delta cap (MAX_DELTA_PER_UPDATE) and a per-tx cooldown
// (MIN_UPDATE_INTERVAL) alone let a patient attacker drift a score
// arbitrarily far by never tripping either check in a single transaction
// (see .claude/ssm_grounding_farming_plan.md §7b).
describe("DragScores cumulative-drift defense", function () {
  async function deployFixture() {
    const accounts = await ethers.getSigners();
    const owner = accounts[0];
    const llmService = accounts[accounts.length - 1];
    const source = accounts[1];

    const DragScores = await ethers.getContractFactory("DragScores", owner);
    const dragScores = await DragScores.deploy(llmService.address);
    await dragScores.waitForDeployment();

    const sourceID = "patient-attack-source";
    await (await dragScores.createScoreRecordByOwner(sourceID, source.address, 10000, 10000, "")).wait();

    return { dragScores, owner, llmService, source, sourceID };
  }

  async function submitUpdate(dragScores, llmService, source, sourceID, newReliability, newUsefulness) {
    const message = `feedback-${Date.now()}-${Math.random()}`;
    const signature = await source.signMessage(message);
    return dragScores
      .connect(llmService)
      .feedbackAndUpdateScoreRecords(
        message,
        [signature],
        [sourceID],
        [newReliability],
        [newUsefulness],
        "info"
      );
  }

  async function advanceSeconds(seconds) {
    await network.provider.send("evm_increaseTime", [seconds]);
    await network.provider.send("evm_mine");
  }

  it("blocks a patient attacker who stays under the per-tx delta/rate caps but accumulates drift within one epoch", async function () {
    const { dragScores, llmService, source, sourceID } = await deployFixture();

    // Mirrors the plan's defense-aware attacker and attack/ssm_score/run_attack.py's
    // own documented default campaign: +4000 per round (just under
    // MAX_DELTA_PER_UPDATE=5000), >2s apart (satisfies MIN_UPDATE_INTERVAL).
    // 4 rounds * 4000 = 16000 = exactly MAX_CUMULATIVE_DELTA_PER_EPOCH.
    let reliability = 10000;
    for (let round = 0; round < 4; round++) {
      await advanceSeconds(3);
      reliability += 4000;
      await expect(
        submitUpdate(dragScores, llmService, source, sourceID, reliability, 10000)
      ).to.not.be.reverted;
    }
    expect(await dragScores.getReliabilityScore(sourceID)).to.equal(26000);

    // The 5th round (the attack's own documented ROUNDS=5 default) would push
    // net drift from the epoch anchor (10000) to 20000 -- over the 16000 cap --
    // even though this single step (+4000) is still individually under
    // MAX_DELTA_PER_UPDATE and the cooldown is respected. This is exactly the
    // scenario the per-tx checks alone miss, and the attack's own prior
    // documented result ("3/3 seeds fully succeeded... R/U +20,000") no
    // longer holds against this contract.
    await advanceSeconds(3);
    await expect(
      submitUpdate(dragScores, llmService, source, sourceID, reliability + 4000, 10000)
    ).to.be.revertedWithCustomError(dragScores, "CumulativeDriftTooLarge");

    // Score must be unchanged by the reverted attempt.
    expect(await dragScores.getReliabilityScore(sourceID)).to.equal(26000);
  });

  it("resets the cumulative cap once the epoch rolls over", async function () {
    const { dragScores, llmService, source, sourceID } = await deployFixture();

    let reliability = 10000;
    for (let round = 0; round < 4; round++) {
      await advanceSeconds(3);
      reliability += 4000;
      await submitUpdate(dragScores, llmService, source, sourceID, reliability, 10000);
    }
    expect(await dragScores.getReliabilityScore(sourceID)).to.equal(26000);

    // Advance past EPOCH_DURATION (1 hour) -- the next update should anchor a
    // fresh epoch at the current score (26000), so another +4000 step is
    // legal again instead of being compared against the original 10000 anchor.
    await advanceSeconds(3601);
    await expect(
      submitUpdate(dragScores, llmService, source, sourceID, reliability + 4000, 10000)
    ).to.not.be.reverted;
    expect(await dragScores.getReliabilityScore(sourceID)).to.equal(30000);
  });

  it("does not interfere with a single legitimate update well under the cumulative cap", async function () {
    const { dragScores, llmService, source, sourceID } = await deployFixture();

    await advanceSeconds(3);
    await expect(
      submitUpdate(dragScores, llmService, source, sourceID, 10050, 10020)
    ).to.not.be.reverted;
    expect(await dragScores.getReliabilityScore(sourceID)).to.equal(10050);
  });
});
