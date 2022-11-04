// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract VoteTest {
    mapping(bytes32 => uint8) public votesReceived;
    bytes32[] public candidateList;


    constructor(bytes32[] memory candidateNames) {
        candidateList = candidateNames;
    }

    function validCandidate(bytes32 candidate) view public returns (bool) {
        for (uint i = 0; i < candidateList.length; i++) {
            if (candidateList[i] == candidate) {
                return true;
            }
        }
        return false;
    }

    function getVoteOfCandidate(bytes32 candidate) view public returns (uint8) {
        require(validCandidate(candidate));
        return votesReceived[candidate];
    }

    function voteToCandidate(bytes32 candidate) public {
        require(validCandidate(candidate));
        votesReceived[candidate] += 1;
    }
}

contract ArrayHandler {

    uint[] arr1;
    uint[] arr2 = new uint[](2);
    byte b1 = "a";
    byte[] b2 = new bytes(3);   //  worse than bytes b3, waste storage, which means
    bytes b3 = new bytes(3);

    function arr1Push() public returns (uint) {
        arr1.push(1);
        arr1.push(6);

        return arr1.length;
    }

    function arr2Length() public view returns (uint) {
        return arr2.length;
    }

    //  if returns array or string, should be define memory
    function arr2() public view returns (uint[] memory) {
        return arr2;
    }

    function arr3Push() public {
        uint[] arr1 = new uint[](2);

        // arr1.push(7);   //  will be error, can't push into an memory array, ever it's a unfixed array
    }
    uint[] a1 = new uint[](6);

    function toDynamic() public pure returns(bytes memory) {
        bytes memory b = new bytes(6);
        uint[] memory a = new uint[](6);
        a[7] = 1;   //  allow set value into a new index which large than array length
        // a.push(222); // No allow push even a unfixed memory array
        // a1.push(222);    //  allow push in a unfixed storage array

        return b;
    }

    string username2 = unicode"爽爽";

    function stringHandle() public pure {
        string memory username = unicode"李荣";
        // username.length; //  don't have length property
        bytes(username).length; //  length is 6, because chinese character occupied 3 bytes


        // fixed bytes to string
        bytes2 c = 0x6768;
        c.length == 2;  //  bytes2 means length is 2, is current?
        bytes memory d = new bytes(c.length);
        for (uint i=0;i<d.length;i++) {
            d[i] = c[i];    //  bypesX index item means a byte as same as bypes index item?
        }
        string(d);
    }

}