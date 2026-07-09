// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "hardhat/console.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

contract DragScores {

    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;
    



    struct ScoreRecord {
        address sourceAddress; // owner's address of the source
        string sourceID; // name of the source
        uint256 timestamp;
        string reserved; // ip address of the source
        int32 reliabilityScore;
        int32 usefulnessScore;
    }

    mapping(string => ScoreRecord) public scoreRecords; // sourceID -> ScoreRecord
    // mapping(address => uint256) public nonces
    mapping(string => bool) public isScoreRecordExists;

    // event ScoreRecordCreated(address indexed sourceAddress, string indexed sourceID, int32 reliabilityScore, int32 usefulnessScore, uint256 timestamp, string info);
    event ScoreRecordUpdated(address indexed sourceAddress, string indexed sourceID, string sourceName, int32 reliabilityScore, int32 usefulnessScore, uint256 timestamp, string info);
    event ScoreRecordUpdatedByOwner(string indexed sourceID, uint256 timestamp, string reserved);
    
    error InvalidSignature();
    error DataSourceNotExists();
    error DataSourceNotFound(string sourceID);
    error InvalidUpdateCount(uint256 numUpdates);

    // ── SSM-Score attack defense (see attack/ssm_score) ────────────────────
    // The attack forges score-update transactions that arbitrarily inflate a
    // source's on-chain reliability/usefulness. These checks bound how much
    // and how often any single feedback transaction can move a score, and
    // restrict who may submit feedback at all.
    error UnauthorizedCaller();
    error ScoreDeltaTooLarge(string sourceID, int32 requestedDelta, int32 cap);
    error UpdateTooFrequent(string sourceID, uint256 secondsRemaining);
    error ScoreOutOfBounds(string sourceID, int32 requestedScore);

    int32 public constant MAX_SCORE = 1_000_000;
    int32 public constant MIN_SCORE = -1_000_000;
    int32 public constant MAX_DELTA_PER_UPDATE = 5_000; // largest single-tx move allowed
    uint256 public constant MIN_UPDATE_INTERVAL = 2; // seconds a source must wait between updates

    address public owner;
    address public llmService; // only address allowed to call feedbackAndUpdateScoreRecords

    constructor(address _llmService) {
        owner = msg.sender;
        llmService = _llmService;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    function setLLMService(address _llmService) public onlyOwner {
        llmService = _llmService;
    }

    function _abs32(int32 x) internal pure returns (int32) {
        return x < 0 ? -x : x;
    }

    function hello() public pure returns (string memory) {
        return "Hello from Ethereum, the service is running!";
    }

    function getReliabilityScore(string memory sourceID) public view returns (int32) {
        return scoreRecords[sourceID].reliabilityScore;
    }

    function getUsefulnessScore(string memory sourceID) public view returns (int32) {
        return scoreRecords[sourceID].usefulnessScore;
    }

    function getScoreRecordsBatch(string[] memory sourceIDs) public view returns (string[] memory returnedSourceIDs, int32[] memory reliabilityScores, int32[] memory usefulnessScores) {
        uint256 numSources = sourceIDs.length;
        returnedSourceIDs = new string[](numSources);
        reliabilityScores = new int32[](numSources);
        usefulnessScores = new int32[](numSources);
        for (uint256 i = 0; i < numSources; i++) {
            returnedSourceIDs[i] = sourceIDs[i];
            reliabilityScores[i] = scoreRecords[sourceIDs[i]].reliabilityScore;
            usefulnessScores[i] = scoreRecords[sourceIDs[i]].usefulnessScore;
        }
        return (returnedSourceIDs, reliabilityScores, usefulnessScores);
    }

    modifier scoreRecordMustExist(string memory sourceID) {
        require(isScoreRecordExists[sourceID], "Score record does not exist");
        _;
    }

    function updateScoreRecordByOwner(
        string memory sourceID,
        uint256 timestamp,
        string memory reserved
    )
        public
        scoreRecordMustExist(sourceID)
        onlyOwner
    {
        ScoreRecord storage scoreRecord = scoreRecords[sourceID];
        scoreRecord.timestamp = timestamp;
        scoreRecord.reserved = reserved;
        emit ScoreRecordUpdatedByOwner(sourceID, timestamp, reserved);
    }

    function createScoreRecordByOwner(
        string memory sourceID,
        address sourceAddress,
        int32 reliabilityScore,
        int32 usefulnessScore,
        string memory reserved
    ) public {
        require(!isScoreRecordExists[sourceID], "Score record already exists");
        scoreRecords[sourceID] = ScoreRecord({
            sourceAddress: sourceAddress,
            sourceID: sourceID,
            timestamp: block.timestamp,
            reserved: reserved,
            reliabilityScore: reliabilityScore,
            usefulnessScore: usefulnessScore
        });
        isScoreRecordExists[sourceID] = true;
    }


    // verify the signatures in batch and update the score records based on the provided updated scords
    function feedbackAndUpdateScoreRecords(
        string memory message, // contains query, selected data sources, the message contained in the signature
        bytes[] memory signatures, // signatures of the data sources
        string[] memory updateSourceIDs, // source IDs to update
        int32[] memory updateReliabilityScores, // reliability scores to update
        int32[] memory updateUsefulnessScores, // usefulness scores to update
        string memory info // info about the feedback
    ) public {
        if (msg.sender != llmService) {
            revert UnauthorizedCaller();
        }
        require(signatures.length == updateSourceIDs.length, "Array lengths must match: signatures and updateSourceIDs");
        require(signatures.length == updateReliabilityScores.length, "Array lengths must match: signatures and updateReliabilityScores");
        require(signatures.length == updateUsefulnessScores.length, "Array lengths must match: signatures and updateUsefulnessScores");

        uint256 timestamp = block.timestamp;
        uint256 numUpdates = signatures.length;
        for (uint256 i = 0; i < numUpdates; i++) {
            string memory sourceID = updateSourceIDs[i];

            if (!isScoreRecordExists[sourceID]) {
                revert DataSourceNotExists();
            }

            bytes32 ethSignedMessageHash = MessageHashUtils.toEthSignedMessageHash(bytes(message));

            // verify the signature
            address recoveredAddress = ethSignedMessageHash.recover(signatures[i]);
            if (recoveredAddress != scoreRecords[sourceID].sourceAddress) {
                revert InvalidSignature();
            }

            ScoreRecord storage scoreRecord = scoreRecords[sourceID];

            // SSM-Score defense: rate limit -- a source's score may only move
            // once per MIN_UPDATE_INTERVAL, which blocks rapid multi-round
            // inflation bursts like the SSM-Score attack's 5-round loop.
            if (timestamp < scoreRecord.timestamp + MIN_UPDATE_INTERVAL) {
                revert UpdateTooFrequent(sourceID, scoreRecord.timestamp + MIN_UPDATE_INTERVAL - timestamp);
            }

            // SSM-Score defense: bound how far a single feedback tx can move
            // reliability/usefulness, so no single (possibly forged) update
            // can catapult a source's score past legitimate feedback noise.
            int32 reliDelta = _abs32(updateReliabilityScores[i] - scoreRecord.reliabilityScore);
            if (reliDelta > MAX_DELTA_PER_UPDATE) {
                revert ScoreDeltaTooLarge(sourceID, reliDelta, MAX_DELTA_PER_UPDATE);
            }
            int32 useDelta = _abs32(updateUsefulnessScores[i] - scoreRecord.usefulnessScore);
            if (useDelta > MAX_DELTA_PER_UPDATE) {
                revert ScoreDeltaTooLarge(sourceID, useDelta, MAX_DELTA_PER_UPDATE);
            }

            // SSM-Score defense: absolute bounds so scores can't be driven to
            // extreme values even by many small, individually-legal updates.
            if (updateReliabilityScores[i] < MIN_SCORE || updateReliabilityScores[i] > MAX_SCORE) {
                revert ScoreOutOfBounds(sourceID, updateReliabilityScores[i]);
            }
            if (updateUsefulnessScores[i] < MIN_SCORE || updateUsefulnessScores[i] > MAX_SCORE) {
                revert ScoreOutOfBounds(sourceID, updateUsefulnessScores[i]);
            }

            scoreRecord.reliabilityScore = updateReliabilityScores[i];
            scoreRecord.usefulnessScore = updateUsefulnessScores[i];
            scoreRecord.timestamp = timestamp;
            emit ScoreRecordUpdated(
                scoreRecord.sourceAddress,
                sourceID, // indexed sourceID
                sourceID, // for event display
                updateReliabilityScores[i],
                updateUsefulnessScores[i],
                timestamp, // block timestamp
                info
            );
        }
        


        // emit FeedbackAndUpdateScoreRecords(message, signatures, updateSourceIDs, updateReliabilityScores, updateUsefulnessScores, info);

        // emit Feedback(query, response, info, nonce);
    }


    function _getMessageHash(
        string memory originalMessage
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(originalMessage));
    }

    function _verifySignature(bytes32 data, bytes memory signature) internal view returns (address) {
        return ECDSA.recover(data, signature);
    }
}