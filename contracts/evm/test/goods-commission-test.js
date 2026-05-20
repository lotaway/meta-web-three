const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("Goods and Commission System", function () {
    let commissionRelation;
    let commissionToken;
    let goodsNFT;
    let owner;
    let buyer;
    let referrer;
    let metaThreeCoin;

    beforeEach(async function () {
        [owner, buyer, referrer] = await ethers.getSigners();

        // 部署 CommissionRelation
        const CommissionRelation = await ethers.getContractFactory("CommissionRelation");
        commissionRelation = await CommissionRelation.deploy();
        await commissionRelation.waitForDeployment();

        // 部署 CommissionToken
        const CommissionToken = await ethers.getContractFactory("CommissionToken");
        commissionToken = await CommissionToken.deploy(
            "Commission Token",
            "CTK",
            await commissionRelation.getAddress()
        );
        await commissionToken.waitForDeployment();

        // 部署 MetaThreeCoin
        const MetaThreeCoin = await ethers.getContractFactory("MetaThreeCoin");
        metaThreeCoin = await MetaThreeCoin.deploy();
        await metaThreeCoin.waitForDeployment();

        // 部署 GoodsNFT
        const GoodsNFT = await ethers.getContractFactory("GoodsNFT");
        goodsNFT = await GoodsNFT.deploy(
            "Goods NFT",
            "GNFT",
            await commissionRelation.getAddress(),
            await metaThreeCoin.getAddress(),
            await commissionToken.getAddress()
        );
        await goodsNFT.waitForDeployment();

        // 授权 GoodsNFT 为 minter
        await commissionRelation.authorizeMinter(await goodsNFT.getAddress());
        await commissionToken.authorizeMinter(await goodsNFT.getAddress());

    });

    it("Should create good, mint with referrer and check commission relation", async function () {
        // 创建商品
        const keys = ["Color", "Size"];
        const values = ["Red", "Large"];
        const price = ethers.parseEther("10");
        await goodsNFT.replenishment("Test Good", keys, values, price);

        // Setup payment for buyer
        await metaThreeCoin.transfer(buyer.address, price);
        await metaThreeCoin.connect(buyer).approve(goodsNFT.getAddress(), price);

        // Mint NFT with referrer
        await goodsNFT.connect(buyer).mint(0, referrer.address);

        // 验证分佣关系是否正确设置
        expect(await commissionRelation.getUpline(buyer.address)).to.equal(referrer.address);
        
        // 验证商品规格
        const specs = await goodsNFT.getGoodSpecifications(0);
        specs.forEach((item, index) => {
            expect(item.key).to.equal(keys[index])
            expect(item.value).to.equal(values[index])
        })

        // 验证 NFT 所有权
        expect(await goodsNFT.ownerOf(0)).to.equal(buyer.address);
    });

    it("Should mint NFT with payment and distribute commissions", async function () {
        // 创建商品
        const price = ethers.parseEther("100"); // 100 MetaThreeCoin
        await goodsNFT.replenishment("Test Good", ["Color"], ["Red"], price);
        
        // 给买家一些 MetaThreeCoin
        await metaThreeCoin.transfer(buyer.address, price);
        await metaThreeCoin.connect(buyer).approve(goodsNFT.getAddress(), price);
        
        // Mint NFT
        await goodsNFT.connect(buyer).mint(0, referrer.address);
        
        // 模拟7天后
        await time.increase(7 * 24 * 60 * 60);

        // 验证佣金分配
        const commission = price * BigInt(10) / BigInt(100); // 10% commission
        const uplineCommission = commission * BigInt(15) / BigInt(100); // 15% upline commission
        
        expect(await commissionToken.balanceOf(buyer.address)).to.equal(commission);
        expect(await commissionToken.balanceOf(referrer.address)).to.equal(uplineCommission);
    });
});