// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Test.sol";
import "../../contracts/GoodsNFT.sol";
import "../../contracts/CommissionRelation.sol";
import "../../contracts/CommissionToken.sol";
import "../../contracts/MetaThreeCoin.sol";

contract GoodsNFTTest is Test {
    MetaThreeCoin private metaThreeCoin;
    CommissionRelation private commissionRelation;
    CommissionToken private commissionToken;
    GoodsNFT private goodsNFT;

    address private buyer = address(0xB0B);
    address private referrer = address(0xA11CE);
    address private dummy = address(0xD00D);

    function setUp() public {
        metaThreeCoin = new MetaThreeCoin("MetaThreeCoin", "M3C");
        commissionRelation = new CommissionRelation();
        commissionToken = new CommissionToken(
            "CommissionToken",
            "CTK",
            address(commissionRelation)
        );
        goodsNFT = new GoodsNFT(
            "GoodsNFT",
            "GNFT",
            address(commissionRelation),
            address(metaThreeCoin),
            address(commissionToken)
        );

        commissionRelation.authorizeMinter(address(goodsNFT));
        commissionToken.authorizeMinter(address(goodsNFT));
    }

    function testMintWithReferrerDistributesCommission() public {
        uint256 price = 1000;
        goodsNFT.replenishment("Good", _single("Color"), _single("Red"), price);

        // Make referrer a valid distributor.
        commissionRelation.setUpline(dummy, referrer);

        metaThreeCoin.transfer(buyer, price);
        vm.startPrank(buyer);
        metaThreeCoin.approve(address(goodsNFT), price);
        goodsNFT.buy(0, referrer);
        vm.stopPrank();

        uint256 commission = (price * goodsNFT.COMMISSION_RATE()) /
            goodsNFT.RATE_DENOMINATOR();
        uint256 uplineCommission = (commission *
            goodsNFT.UPLINE_COMMISSION_RATE()) / goodsNFT.RATE_DENOMINATOR();

        assertEq(commissionToken.balanceOf(buyer), commission);
        assertEq(commissionToken.balanceOf(referrer), uplineCommission);
    }

    function testFuzz_CommissionMath(uint96 price) public {
        vm.assume(price > 0);
        goodsNFT.replenishment("Good", _single("Color"), _single("Red"), price);
        commissionRelation.setUpline(dummy, referrer);

        metaThreeCoin.transfer(buyer, uint256(price));
        vm.startPrank(buyer);
        metaThreeCoin.approve(address(goodsNFT), uint256(price));
        goodsNFT.buy(0, referrer);
        vm.stopPrank();

        uint256 commission = (uint256(price) *
            goodsNFT.COMMISSION_RATE()) / goodsNFT.RATE_DENOMINATOR();
        uint256 uplineCommission = (commission *
            goodsNFT.UPLINE_COMMISSION_RATE()) / goodsNFT.RATE_DENOMINATOR();

        assertEq(commissionToken.balanceOf(buyer), commission);
        assertEq(commissionToken.balanceOf(referrer), uplineCommission);
    }

    function _single(
        string memory value
    ) private pure returns (string[] memory arr) {
        arr = new string[](1);
        arr[0] = value;
    }
}
