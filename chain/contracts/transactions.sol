//  SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Transactions {
    uint256 transactionCounter;

    event Transfer(address from, address receiver, uint amount, string message, uint256 timestamp, string keyword);

    struct TransferStruct {
        address sender;
        address receiver;
        uint amount;
        string message;
        uint256 timestamp;
        string keyword;
    }

    TransferStruct[] transferStruct;

    function addRecord(address payable receiver, uint amount, string memory message, string memory keyword) public {
        transferStruct.push(TransferStruct({
            sender: msg.sender,
            receiver: receiver,
            amount: amount,
            message: message,
            timestamp: block.timestamp,
            keyword: keyword
        }));
        emit Transfer(msg.sender, receiver, amount, message, block.timestamp, keyword);
    }

    function getRecord() public view returns (TransferStruct[] memory) {
        return transferStruct;
    }

    function getRecordCount() public view returns (uint256) {
        return transferStruct.length;
    }
}