const { ethers } = require("hardhat");
const { expect } = require("chai");

describe("ERC20 Permit Test", function () {
  let owner, spender, other;
  let token;
  
  before(async function () {
    [owner, spender, other] = await ethers.getSigners();
    
    const Token = await ethers.getContractFactory("MyTokenWithPermit");
    token = await Token.deploy();
    await token.deployed();
  });

  it("Should permit and transfer", async function () {
    const amount = ethers.utils.parseEther("100");
    const deadline = Math.floor(Date.now() / 1000) + 3600; // 1小时后过期

    // 1. 获取当前nonce
    const nonce = await token.nonces(owner.address);
    
    // 2. 准备EIP-712签名数据
    const domain = {
      name: await token.name(),
      version: "1",
      chainId: await owner.getChainId(),
      verifyingContract: token.address,
    };

    const types = {
      Permit: [
        { name: "owner", type: "address" },
        { name: "spender", type: "address" },
        { name: "value", type: "uint256" },
        { name: "nonce", type: "uint256" },
        { name: "deadline", type: "uint256" },
      ],
    };

    const values = {
      owner: owner.address,
      spender: spender.address,
      value: amount,
      nonce: nonce,
      deadline: deadline,
    };

    // 3. 离线签名（不发送交易）
    const signature = await owner._signTypedData(domain, types, values);
    const { v, r, s } = ethers.utils.splitSignature(signature);

    // 4. 验证初始授权为0
    expect(await token.allowance(owner.address, spender.address)).to.equal(0);

    // 5. spender使用签名调用permit
    await token.connect(spender).permit(
      owner.address,
      spender.address,
      amount,
      deadline,
      v, r, s
    );

    // 6. 验证授权成功
    expect(await token.allowance(owner.address, spender.address)).to.equal(amount);

    // 7. spender使用授权金额转账
    await token.connect(spender).transferFrom(
      owner.address,
      other.address,
      amount
    );

    // 8. 验证转账成功
    expect(await token.balanceOf(other.address)).to.equal(amount);
  });
});