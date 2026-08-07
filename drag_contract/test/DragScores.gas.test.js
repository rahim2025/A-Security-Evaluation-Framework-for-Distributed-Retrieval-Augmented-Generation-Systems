const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("DragScores gas measurements", function () {
  async function deployFixture() {
    const accounts = await ethers.getSigners();
    const owner = accounts[0];
    const llmService = accounts[accounts.length - 1];
    const signers = accounts.slice(0, accounts.length - 1); // reserve last account as llmService
    const DragScores = await ethers.getContractFactory("DragScores", owner);
    const dragScores = await DragScores.deploy(llmService.address);
    await dragScores.waitForDeployment();

    // Seed one score record per (batchSize, index) pair used below so each
    // source is only ever updated once -- avoids the SSM-Score defense's
    // per-source cooldown between calls to feedbackAndUpdateScoreRecords.
    const maxN = Math.min(20, signers.length);
    const batchSizes = [1, 2, 5, 10, Math.min(15, maxN), maxN];
    for (const n of batchSizes) {
      for (let i = 0; i < n; i++) {
        const sourceID = `source-b${n}-${i}`;
        const sourceAddress = signers[i % signers.length].address;
        const tx = await dragScores.createScoreRecordByOwner(
          sourceID,
          sourceAddress,
          0,
          0,
          ""
        );
        await tx.wait();
      }
    }

    return { dragScores, owner, llmService, signers, batchSizes };
  }

  async function measureBatchGas(dragScores, llmService, signers, batchSize) {
    if (batchSize > signers.length) {
      batchSize = signers.length;
    }
    const message = `feedback-${Date.now()}`;
    const info = `info-${batchSize}`;

    const signatures = [];
    const updateSourceIDs = [];
    const updateReliabilityScores = [];
    const updateUsefulnessScores = [];

    for (let i = 0; i < batchSize; i++) {
      const srcId = `source-b${batchSize}-${i}`;
      const signer = signers[i % signers.length];
      const signature = await signer.signMessage(message);

      signatures.push(signature);
      updateSourceIDs.push(srcId);
      updateReliabilityScores.push(10 + i);
      updateUsefulnessScores.push(20 + i);
    }

    // estimate gas, then execute and get actual used gas
    const gasEstimate = await dragScores
      .connect(llmService)
      .feedbackAndUpdateScoreRecords.estimateGas(
        message,
        signatures,
        updateSourceIDs,
        updateReliabilityScores,
        updateUsefulnessScores,
        info
      );

    const tx = await dragScores
      .connect(llmService)
      .feedbackAndUpdateScoreRecords(
        message,
        signatures,
        updateSourceIDs,
        updateReliabilityScores,
        updateUsefulnessScores,
        info
      );
    const receipt = await tx.wait();

    return { gasEstimate: gasEstimate, gasUsed: receipt.gasUsed };
  }

  it("measures gas for varying batch sizes", async function () {
    const { dragScores, llmService, signers, batchSizes } = await deployFixture();

    for (const n of batchSizes) {
      const { gasEstimate, gasUsed } = await measureBatchGas(
        dragScores,
        llmService,
        signers,
        n
      );

      // basic correctness: no reverts and scores updated
      for (let i = 0; i < n; i++) {
        const srcId = `source-b${n}-${i}`;
        const reliability = await dragScores.getReliabilityScore(srcId);
        const usefulness = await dragScores.getUsefulnessScore(srcId);
        expect(reliability).to.equal(10 + i);
        expect(usefulness).to.equal(20 + i);
      }

      // Log gas numbers for inspection
      console.log(
        `feedbackAndUpdateScoreRecords: n=${n}, estimate=${gasEstimate.toString()}, used=${gasUsed.toString()}, perUpdate≈${gasUsed / BigInt(n)}`
      );
    }
  });
});


