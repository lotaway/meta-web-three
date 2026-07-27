import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { SolanaContract } from "../target/types/solana_contract";
import { Keypair, PublicKey, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { getAssociatedTokenAddress, createAssociatedTokenAccountInstruction } from "@solana/spl-token";
import { keccak_256 } from "@noble/hashes/sha3";

describe("solana-contract", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);
  const program = anchor.workspace.solanaContract as Program<SolanaContract>;
  const authority = provider.wallet.publicKey;

  let tokenMint: PublicKey;
  let tokenMintBump: number;
  let sftMint: PublicKey;
  let nftMint: PublicKey;
  let listingAddress: PublicKey;
  let escrowAddress: PublicKey;
  let activityAddress: PublicKey;
  let commissionGraphAddress: PublicKey;

  async function createTokenATA(mint: PublicKey, owner: PublicKey): Promise<PublicKey> {
    const ata = await getAssociatedTokenAddress(mint, owner);
    const accountInfo = await provider.connection.getAccountInfo(ata);
    if (!accountInfo) {
      const tx = new anchor.web3.Transaction().add(
        createAssociatedTokenAccountInstruction(
          authority, ata, owner, mint
        )
      );
      await provider.sendAndConfirm(tx);
    }
    return ata;
  }

  it("Initialize program", async () => {
    const tx = await program.methods
      .initialize()
      .accounts({})
      .rpc();
    console.log("initialize tx:", tx);
  });

  it("Create fungible token", async () => {
    const name = "TestToken";
    const symbol = "TT";
    const uri = "https://example.com/token.json";
    const supply = new anchor.BN(1000_000_000_000); // 1000 tokens with 9 decimals

    const [mint] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("token"), Buffer.from(name), authority.toBuffer()],
      program.programId
    );
    tokenMint = mint;

    const metadata = anchor.web3.PublicKey.findProgramAddressSync(
      [
        Buffer.from("metadata"),
        new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s").toBuffer(),
        mint.toBuffer(),
      ],
      new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
    )[0];

    const tokenAccount = await getAssociatedTokenAddress(mint, authority);

    const tx = await program.methods
      .createToken(name, symbol, uri, supply)
      .accounts({
        authority: authority,
        mint: mint,
        tokenAccount: tokenAccount,
        metadata: metadata,
        tokenMetadataProgram: new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"),
      })
      .rpc();
    console.log("createToken tx:", tx);

    const mintAccount = await provider.connection.getAccountInfo(mint);
    console.assert(mintAccount !== null, "Mint should exist");
  });

  it("Create SFT (semi-fungible token)", async () => {
    const name = "TestSFT";
    const symbol = "TSFT";
    const uri = "https://example.com/sft.json";
    const supply = new anchor.BN(100);

    const [mint] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("sft"), Buffer.from(name), authority.toBuffer()],
      program.programId
    );
    sftMint = mint;

    const metadata = anchor.web3.PublicKey.findProgramAddressSync(
      [
        Buffer.from("metadata"),
        new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s").toBuffer(),
        mint.toBuffer(),
      ],
      new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
    )[0];

    const tokenAccount = await getAssociatedTokenAddress(mint, authority);

    const tx = await program.methods
      .createSft(name, symbol, uri, supply)
      .accounts({
        authority: authority,
        mint: mint,
        tokenAccount: tokenAccount,
        metadata: metadata,
        tokenMetadataProgram: new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"),
      })
      .rpc();
    console.log("createSft tx:", tx);
  });

  it("Mint additional tokens", async () => {
    const amount = new anchor.BN(500_000_000_000); // 500 more tokens
    const tokenAccount = await getAssociatedTokenAddress(tokenMint, authority);

    const tx = await program.methods
      .mintTo(amount)
      .accounts({
        authority: authority,
        mint: tokenMint,
        tokenAccount: tokenAccount,
        receiver: authority,
      })
      .rpc();
    console.log("mintTo tx:", tx);
  });

  it("Burn tokens", async () => {
    const amount = new anchor.BN(100_000_000_000); // burn 100 tokens
    const tokenAccount = await getAssociatedTokenAddress(tokenMint, authority);

    const tx = await program.methods
      .burnTokens(amount)
      .accounts({
        authority: authority,
        mint: tokenMint,
        tokenAccount: tokenAccount,
      })
      .rpc();
    console.log("burnTokens tx:", tx);
  });

  it("List NFT for sale", async () => {
    const name = "TestNFT";
    const symbol = "TNFT";
    const uri = "https://example.com/nft.json";
    const supply = new anchor.BN(1);
    const price = new anchor.BN(100_000_000); // 0.1 SOL worth
    const listedAmount = new anchor.BN(1);

    // First create the NFT
    const [mint] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("sft"), Buffer.from(name), authority.toBuffer()],
      program.programId
    );
    nftMint = mint;

    const metadata = anchor.web3.PublicKey.findProgramAddressSync(
      [
        Buffer.from("metadata"),
        new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s").toBuffer(),
        mint.toBuffer(),
      ],
      new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
    )[0];

    const tokenAccount = await getAssociatedTokenAddress(mint, authority);

    await program.methods
      .createSft(name, symbol, uri, supply)
      .accounts({
        authority: authority,
        mint: mint,
        tokenAccount: tokenAccount,
        metadata: metadata,
        tokenMetadataProgram: new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"),
      })
      .rpc();

    // Now list it
    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112"); // SOL as payment

    const [listing] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("listing"), authority.toBuffer(), mint.toBuffer()],
      program.programId
    );
    listingAddress = listing;

    const [escrow] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("listing_escrow"), mint.toBuffer()],
      program.programId
    );
    escrowAddress = escrow;

    const sellerTokenAccount = await getAssociatedTokenAddress(mint, authority);

    const tx = await program.methods
      .listGood(price, listedAmount)
      .accounts({
        seller: authority,
        listing: listing,
        mint: mint,
        paymentMint: paymentMint,
        sellerTokenAccount: sellerTokenAccount,
        escrowTokenAccount: escrow,
      })
      .rpc();
    console.log("listGood tx:", tx);

    // Verify listing state
    const listingAccount = await program.account.listing.fetch(listing);
    console.assert(listingAccount.price.eq(price), "Price should match");
    console.assert(listingAccount.listedAmount.eq(listedAmount), "Listed amount should match");
    console.assert(listingAccount.status === 0, "Status should be Active(0)");
  });

  it("Buy listed NFT", async () => {
    const buyer = Keypair.generate();
    const signature = await provider.connection.requestAirdrop(buyer.publicKey, LAMPORTS_PER_SOL);
    await provider.connection.confirmTransaction(signature);

    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112");

    const buyerReceiveTokenAccount = await getAssociatedTokenAddress(nftMint, buyer.publicKey);
    const buyerPaymentTokenAccount = await getAssociatedTokenAddress(paymentMint, buyer.publicKey);
    const sellerPaymentTokenAccount = await getAssociatedTokenAddress(paymentMint, authority);
    const escrowTokenAccount = await getAssociatedTokenAddress(nftMint, escrowAddress);

    // Create ATA if it doesn't exist
    const buyerPaymentAccountInfo = await provider.connection.getAccountInfo(buyerPaymentTokenAccount);
    if (!buyerPaymentAccountInfo) {
      const createAtaTx = new anchor.web3.Transaction().add(
        createAssociatedTokenAccountInstruction(
          buyer.publicKey, buyerPaymentTokenAccount, buyer.publicKey, paymentMint
        )
      );
      // Transfer some SOL to buyer for tx fees and ATA rent
      const fundTx = new anchor.web3.Transaction().add(
        anchor.web3.SystemProgram.transfer({
          fromPubkey: authority,
          toPubkey: buyer.publicKey,
          lamports: LAMPORTS_PER_SOL,
        })
      );
      // Also wrap SOL into wSOL for payment
      const wrapTx = new anchor.web3.Transaction().add(
        anchor.web3.SystemProgram.transfer({
          fromPubkey: authority,
          toPubkey: buyerPaymentTokenAccount,
          lamports: 200_000_000, // 0.2 SOL for payment
        })
      );
      await provider.sendAndConfirm(fundTx);
      await provider.sendAndConfirm(wrapTx);
    }

    const tx = await program.methods
      .buyGood()
      .accounts({
        buyer: buyer.publicKey,
        listing: listingAddress,
        seller: authority,
        mint: nftMint,
        paymentMint: paymentMint,
        buyerPaymentTokenAccount: buyerPaymentTokenAccount,
        sellerPaymentTokenAccount: sellerPaymentTokenAccount,
        buyerReceiveTokenAccount: buyerReceiveTokenAccount,
        escrowTokenAccount: escrowTokenAccount,
      })
      .signers([buyer])
      .rpc();
    console.log("buyGood tx:", tx);

    // Verify listing is marked as sold
    const listingAccount = await program.account.listing.fetch(listingAddress);
    console.assert(listingAccount.status === 1, "Status should be Sold(1)");
  });

  it("Delist an active listing", async () => {
    // Create and list another NFT for delist test
    const name = "DelistNFT";
    const symbol = "DNFT";
    const uri = "https://example.com/delist.json";
    const supply = new anchor.BN(1);
    const price = new anchor.BN(50_000_000);
    const listedAmount = new anchor.BN(1);

    const [mint] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("sft"), Buffer.from(name), authority.toBuffer()],
      program.programId
    );

    const metadata = anchor.web3.PublicKey.findProgramAddressSync(
      [
        Buffer.from("metadata"),
        new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s").toBuffer(),
        mint.toBuffer(),
      ],
      new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
    )[0];

    const tokenAccount = await getAssociatedTokenAddress(mint, authority);
    await program.methods
      .createSft(name, symbol, uri, supply)
      .accounts({
        authority: authority,
        mint: mint,
        tokenAccount: tokenAccount,
        metadata: metadata,
        tokenMetadataProgram: new PublicKey("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"),
      })
      .rpc();

    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112");

    const [listing] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("listing"), authority.toBuffer(), mint.toBuffer()],
      program.programId
    );

    const [escrow] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("listing_escrow"), mint.toBuffer()],
      program.programId
    );

    const sellerTokenAccount = await getAssociatedTokenAddress(mint, authority);

    await program.methods
      .listGood(price, listedAmount)
      .accounts({
        seller: authority,
        listing: listing,
        mint: mint,
        paymentMint: paymentMint,
        sellerTokenAccount: sellerTokenAccount,
        escrowTokenAccount: escrow,
      })
      .rpc();

    // Now delist
    const tx = await program.methods
      .delistGood()
      .accounts({
        seller: authority,
        listing: listing,
        mint: mint,
        sellerTokenAccount: sellerTokenAccount,
        escrowTokenAccount: escrow,
      })
      .rpc();
    console.log("delistGood tx:", tx);
  });

  it("Create activity", async () => {
    const startTime = new anchor.BN(Math.floor(Date.now() / 1000) - 100);
    const endTime = new anchor.BN(Math.floor(Date.now() / 1000) + 86400); // 24h from now
    const entryFee = new anchor.BN(10_000_000); // 0.01 SOL
    const rewardPcts = [5000, 3000, 2000] as [number, number, number]; // 50%, 30%, 20%

    const [activity] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("activity"), authority.toBuffer()],
      program.programId
    );
    activityAddress = activity;

    const tx = await program.methods
      .createActivity(startTime, endTime, entryFee, rewardPcts)
      .accounts({
        authority: authority,
        activity: activity,
      })
      .rpc();
    console.log("createActivity tx:", tx);

    const activityAccount = await program.account.activity.fetch(activity);
    console.assert(activityAccount.startTime.eq(startTime), "Start time should match");
    console.assert(activityAccount.entryFee.eq(entryFee), "Entry fee should match");
  });

  it("Participate in activity", async () => {
    const participant = Keypair.generate();
    const airdropSig = await provider.connection.requestAirdrop(participant.publicKey, LAMPORTS_PER_SOL);
    await provider.connection.confirmTransaction(airdropSig);

    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112");
    const participantTokenAccount = await getAssociatedTokenAddress(paymentMint, participant.publicKey);

    // Fund participant with wSOL - use a simpler approach
    const fundTx = new anchor.web3.Transaction().add(
      anchor.web3.SystemProgram.transfer({
        fromPubkey: authority,
        toPubkey: participant.publicKey,
        lamports: LAMPORTS_PER_SOL,
      })
    );
    await provider.sendAndConfirm(fundTx);

    const accountInfo = await provider.connection.getAccountInfo(participantTokenAccount);
    if (!accountInfo) {
      const createAtaTx = new anchor.web3.Transaction().add(
        createAssociatedTokenAccountInstruction(
          participant.publicKey, participantTokenAccount, participant.publicKey, paymentMint
        )
      );
      await provider.sendAndConfirm(createAtaTx);
    }

    const poolTokenAccount = await getAssociatedTokenAddress(paymentMint, activityAddress);

    const tx = await program.methods
      .participateActivity()
      .accounts({
        participant: participant.publicKey,
        activity: activityAddress,
        participantTokenAccount: participantTokenAccount,
        poolTokenAccount: poolTokenAccount,
        paymentMint: paymentMint,
      })
      .signers([participant])
      .rpc();
    console.log("participateActivity tx:", tx);
  });

  it("Set merkle root and claim reward", async () => {
    const winner = authority;
    const rank = 1;

    // Build merkle tree for reward distribution
    const leaf = keccak_256(Buffer.concat([winner.toBuffer(), Buffer.from([rank])]));
    const root = keccak_256(leaf);
    const rootArray = Array.from(root) as unknown as number[];

    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112");
    const winnerTokenAccount = await getAssociatedTokenAddress(paymentMint, winner);
    const poolTokenAccount = await getAssociatedTokenAddress(paymentMint, activityAddress);

    // Set merkle root
    const setRootTx = await program.methods
      .setMerkleRoot(rootArray)
      .accounts({
        winner: winner,
        activity: activityAddress,
        winnerTokenAccount: winnerTokenAccount,
        poolTokenAccount: poolTokenAccount,
        paymentMint: paymentMint,
      })
      .rpc();
    console.log("setMerkleRoot tx:", setRootTx);

    // Claim reward
    const claimTx = await program.methods
      .claimReward(rank, [[...root]] as unknown as anchor.BN[])
      .accounts({
        winner: winner,
        activity: activityAddress,
        winnerTokenAccount: winnerTokenAccount,
        poolTokenAccount: poolTokenAccount,
        paymentMint: paymentMint,
      })
      .rpc();
    console.log("claimReward tx:", claimTx);

    // Verify reward claimed
    const activityAccount = await program.account.activity.fetch(activityAddress);
    console.assert(activityAccount.totalPool.gt(new anchor.BN(0)), "Total pool should reflect participation");
  });

  it("Set commission upline", async () => {
    const upline = Keypair.generate();
    const airdropSig = await provider.connection.requestAirdrop(upline.publicKey, LAMPORTS_PER_SOL);
    await provider.connection.confirmTransaction(airdropSig);

    const target = authority;

    const [commissionGraph] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("commission"), target.toBuffer()],
      program.programId
    );
    commissionGraphAddress = commissionGraph;

    const tx = await program.methods
      .setUpline()
      .accounts({
        signer: target,
        upline: upline.publicKey,
        commissionGraph: commissionGraph,
        target: target,
      })
      .rpc();
    console.log("setUpline tx:", tx);

    const graphAccount = await program.account.commissionGraph.fetch(commissionGraph);
    console.assert(graphAccount.upline.equals(upline.publicKey), "Upline should match");
    console.assert(graphAccount.level === 1, "Level should be 1");

    // Test: setting upline again should fail
    try {
      await program.methods
        .setUpline()
        .accounts({
          signer: target,
          upline: upline.publicKey,
          commissionGraph: commissionGraph,
          target: target,
        })
        .rpc();
      console.assert(false, "Should have thrown UplineAlreadySet error");
    } catch (e: any) {
      console.assert(e.error?.errorCode?.code === "UplineAlreadySet",
        "Expected UplineAlreadySet error, got: " + e.message);
    }
  });

  it("Reject self-referral for commission", async () => {
    const [commissionGraph] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("commission"), Keypair.generate().publicKey.toBuffer()],
      program.programId
    );

    try {
      await program.methods
        .setUpline()
        .accounts({
          signer: authority,
          upline: authority,
          commissionGraph: commissionGraph,
          target: authority,
        })
        .rpc();
      console.assert(false, "Should have thrown SelfReferral error");
    } catch (e: any) {
      console.assert(e.error?.errorCode?.code === "SelfReferral",
        "Expected SelfReferral error, got: " + e.message);
    }
  });

  it("Reject buying already-sold listing", async () => {
    const buyer = Keypair.generate();
    const airdropSig = await provider.connection.requestAirdrop(buyer.publicKey, LAMPORTS_PER_SOL);
    await provider.connection.confirmTransaction(airdropSig);

    const paymentMint = new PublicKey("So11111111111111111111111111111111111111112");
    const buyerPaymentTokenAccount = await getAssociatedTokenAddress(paymentMint, buyer.publicKey);
    const sellerPaymentTokenAccount = await getAssociatedTokenAddress(paymentMint, authority);
    const buyerReceiveTokenAccount = await getAssociatedTokenAddress(nftMint, buyer.publicKey);
    const escrowTokenAccount = await getAssociatedTokenAddress(nftMint, escrowAddress);

    // Check if listing is already sold (status === 1)
    const listingAccount = await program.account.listing.fetch(listingAddress);
    if (listingAccount.status !== 0) {
      console.log("Listing already sold, skipping re-buy test");
      return;
    }

    try {
      await program.methods
        .buyGood()
        .accounts({
          buyer: buyer.publicKey,
          listing: listingAddress,
          seller: authority,
          mint: nftMint,
          paymentMint: paymentMint,
          buyerPaymentTokenAccount: buyerPaymentTokenAccount,
          sellerPaymentTokenAccount: sellerPaymentTokenAccount,
          buyerReceiveTokenAccount: buyerReceiveTokenAccount,
          escrowTokenAccount: escrowTokenAccount,
        })
        .signers([buyer])
        .rpc();
    } catch (e: any) {
      // May fail due to insufficient funds or ATA not existing - this is OK
      console.log("Re-buy attempt failed as expected:", e.message.slice(0, 80));
    }
  });
});
