const {expect} = require("chai");
const {ethers} = require("hardhat");
const {time} = require("@nomicfoundation/hardhat-network-helpers");

describe("Integration flow: goods + activity", function () {
    let owner;
    let buyer;
    let referrer;
    let commissionRelation;
    let commissionToken;
    let metaThreeCoin;
    let goodsNFT;
    let activityFactory;

    beforeEach(async function () {
        [owner, buyer, referrer] = await ethers.getSigners();

        const CommissionRelation = await ethers.getContractFactory("CommissionRelation");
        commissionRelation = await CommissionRelation.deploy();
        await commissionRelation.waitForDeployment();

        const CommissionToken = await ethers.getContractFactory("CommissionToken");
        commissionToken = await CommissionToken.deploy(
            "CommissionToken",
            "CTK",
            await commissionRelation.getAddress()
        );
        await commissionToken.waitForDeployment();

        const MetaThreeCoin = await ethers.getContractFactory("MetaThreeCoin");
        metaThreeCoin = await MetaThreeCoin.deploy("MetaThreeCoin", "M3C");
        await metaThreeCoin.waitForDeployment();

        const GoodsNFT = await ethers.getContractFactory("GoodsNFT");
        goodsNFT = await GoodsNFT.deploy(
            "GoodsNFT",
            "GNFT",
            await commissionRelation.getAddress(),
            await metaThreeCoin.getAddress(),
            await commissionToken.getAddress()
        );
        await goodsNFT.waitForDeployment();

        await commissionRelation.authorizeMinter(await goodsNFT.getAddress());
        await commissionToken.authorizeMinter(await goodsNFT.getAddress());

        const ActivityFactory = await ethers.getContractFactory("ActivityFactory");
        activityFactory = await ActivityFactory.deploy(await metaThreeCoin.getAddress());
        await activityFactory.waitForDeployment();
    });

    it("User purchases goods and claims activity reward", async function () {
        const price = 1000n;
        await goodsNFT.replenishment("Good", ["Color"], ["Red"], price);

        // Ensure referrer is a valid distributor.
        await commissionRelation.setUpline(buyer.address, referrer.address);

        await metaThreeCoin.transfer(buyer.address, price);
        await metaThreeCoin.connect(buyer).approve(await goodsNFT.getAddress(), price);
        await goodsNFT.connect(buyer).buy(0, referrer.address);

        expect(await goodsNFT.ownerOf(0)).to.equal(buyer.address);

        const now = await time.latest();
        const startTime = now + 10;
        const endTime = now + 3600;
        await activityFactory.createActivity(
            startTime,
            endTime,
            [50, 30, 20],
            100
        );

        const activities = await activityFactory.getActivities();
        const activity = await ethers.getContractAt("Activity", activities[0]);

        await time.increaseTo(startTime + 1);
        await metaThreeCoin.connect(buyer).approve(await activity.getAddress(), 100);
        await activity.connect(buyer).participate();

        // Prepare a 1-leaf merkle tree: root == leaf, proof == []
        const leaf = ethers.solidityPackedKeccak256(
            ["address", "uint256"],
            [buyer.address, 1]
        );
        await activity.setMerkleRoot(leaf);

        await activity.connect(buyer).claimReward(1, []);
        expect(await metaThreeCoin.balanceOf(buyer.address)).to.be.gt(0);
    });

    it("Non-owner cannot create activity", async function () {
        await expect(
            activityFactory.connect(buyer).createActivity(
                1,
                2,
                [50, 30, 20],
                100
            )
        ).to.be.reverted;
    });
});
