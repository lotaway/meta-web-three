import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { SolanaContract } from "../target/types/solana_contract";
import { Keypair, PublicKey } from "@solana/web3.js";
import { keypairIdentity, Metaplex } from "@metaplex-foundation/js";

describe("solana-contract", () => {
  // Configure the client to use the local cluster.

  let program: Program<SolanaContract>
  let createdNFT

  before(async () => {
    anchor.setProvider(anchor.AnchorProvider.env());

    program = anchor.workspace.solanaContract as Program<SolanaContract>;

    const tx = await program.methods
      .initialize()
      .accounts({
        tokenMintAccount: PublicKey.findProgramAddressSync([
          Buffer.from("token_manager"),
        ], program.programId)[0],
        // signer
      })
      .rpc();
    console.log("Your transaction signature", tx);
  });

  it("Is initialized!", async () => {
    // Add your test here.
  });

  it("Is nft created", async () => {
    // const tx = await program.methods.createNft().rpc();
    const metaplex = new Metaplex(anchor.getProvider().connection).use(
      keypairIdentity(anchor.getProvider().wallet.payer)
    );
    createdNFT = await metaplex.nfts().create({
      name: "Test NFT",
      symbol: "TEST",
      uri: "https://example.com/nft.json",
      sellerFeeBasisPoints: 0,
    });
    console.log("Your transaction signature", createdNFT);
  });
});
