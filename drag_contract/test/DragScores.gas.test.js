const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("DragScores gas measurements", function () {
  async function deployFixture() {
    const accounts = await ethers.getSigners();
    const owner = accounts[0];
    const signers = accounts; // use all accounts to avoid running out
    const DragScores = await ethers.getContractFactory("DragScores", owner);
    const dragScores = await DragScores.deploy();
    await dragScores.waitForDeployment();

    // seed N score records owned by owner, with sourceAddress set to specific signers
    const numSeed = Math.min(20, signers.length); // ensure within bounds
    for (let i = 0; i < numSeed; i++) {
      const sourceID = `source-${i}`;
      const sourceAddress = signers[i].address;
      const reliability = 0;
      const usefulness = 0;
      const reserved = "";
      const tx = await dragScores.createScoreRecordByOwner(
        sourceID,
        sourceAddress,
        reliability,
        usefulness,
        reserved
      );
      await tx.wait();
    }

    return { dragScores, owner, signers };
  }

  async function measureBatchGas(dragScores, signers, batchSize) {
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
      const srcId = `source-${i}`;
      const signer = signers[i];
      const signature = await signer.signMessage(message);

      signatures.push(signature);
      updateSourceIDs.push(srcId);
      updateReliabilityScores.push(10 + i);
      updateUsefulnessScores.push(20 + i);
    }

    // estimate gas, then execute and get actual used gas
    const gasEstimate = await dragScores
      .connect(signers[0])
      .feedbackAndUpdateScoreRecords.estimateGas(
        message,
        signatures,
        updateSourceIDs,
        updateReliabilityScores,
        updateUsefulnessScores,
        info
      );

    const tx = await dragScores
      .connect(signers[0])
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
    const { dragScores, signers } = await deployFixture();

    const maxN = Math.min(20, signers.length);
    const batchSizes = [1, 2, 5, 10, Math.min(15, maxN), maxN];
    for (const n of batchSizes) {
      const { gasEstimate, gasUsed } = await measureBatchGas(
        dragScores,
        signers,
        n
      );

      // basic correctness: no reverts and scores updated
      for (let i = 0; i < n; i++) {
        const srcId = `source-${i}`;
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


