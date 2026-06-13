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
    
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
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