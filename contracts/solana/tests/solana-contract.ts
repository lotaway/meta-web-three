import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { SolanaContract } from "../target/types/solana_contract";
import { Keypair, PublicKey } from "@solana/web3.js";
import { CreateCompressedNftOutput, keypairIdentity, Metaplex } from "@metaplex-foundation/js";

describe("solana-contract", () => {
  // Configure the client to use the local cluster.

  let program: Program<SolanaContract>
  let createdNFT: CreateCompressedNftOutput
  let createNFTKeypair: Keypair

  before(async () => {
    anchor.setProvider(anchor.AnchorProvider.env())
    program = anchor.workspace.solanaContract as Program<SolanaContract>
  })

  it("Is nft created", async () => {
    // const tx = await program.methods.createNft().rpc();
    createNFTKeypair = Keypair.generate()
    const metaplex = new Metaplex(anchor.getProvider().connection).use(
      keypairIdentity(anchor.getProvider().wallet.payer)
    )
    if (metaplex.cluster === "localnet") {
      createdNFT = {
        mintAddress: createNFTKeypair.publicKey
      } as CreateCompressedNftOutput
      console.log("createdNFT", createdNFT.mintAddress)
      return
    }
    createdNFT = await metaplex.nfts().create({
      name: "Test NFT",
      symbol: "TEST",
      uri: "https://example.com/nft.json",
      sellerFeeBasisPoints: 0,
      useExistingMint: createNFTKeypair.publicKey,
    })
    console.log("Your transaction signature", createdNFT)
  })

  it("Is initialized!", async () => {
    const tx = await program.methods
      .initialize()
      .accounts({
        tokenMintAccount: createNFTKeypair.publicKey.toBase58(),
      })
      .signers([createNFTKeypair])
      .rpc()
    console.log("Your transaction signature", tx)
  })

  it("Is nft transfer", async () => {
  })
})
