use std::env;
use dotenv::dotenv;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::dispatching::dialogue::InMemStorage;
use teloxide::dispatching::HandlerExt;
use teloxide::types::Update;
use dptree::prelude::*;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo};
use teloxide::utils::command::{BotCommands};

#[derive(BotCommands, PartialEq, Debug)]
#[command(rename_rule = "lowercase", description = "These commands are supported:")]
enum UserCommandType {
    #[command(description = "Display the help list")]
    Help,
    #[command(description = "Start the App")]
    Start,
}

pub async fn run() {
    // teloxide::handler!();
    log::info!("Starting telegram bot...");

    dotenv().ok();
    let mut bot_name = env::var("BOT_NAME").expect("Can't found BOT_NAME");
    // need env TELOXIDE_TOKEN
    let bot = Bot::from_env();
    Dispatcher::builder(bot, dptree::entry()
        .branch(
            Update::filter_message()
                .branch(
                    dptree::filter(|msg: Message| msg.text().is_some())
                        .endpoint(|msg: Message, bot: Bot| async {
                            if let Some(text) = msg.text() {
                                if text.starts_with('/') {
                                    match UserCommandType::parse(text, bot_name.clone().as_ref()) {
                                        Ok(userCommand) => {
                                            answer(bot, msg, userCommand).await?;
                                        }
                                        Err(err) => {
                                            println!("Failed to parse command: {}", err);
                                        }
                                    }
                                }
                            }
                            Result::<(), teloxide::RequestError>::Ok(())
                        }),
                ),
        )).build().dispatch().await;
}

async fn answer(bot: Bot, msg: Message, cmd: UserCommandType) -> ResponseResult<()> {
    match cmd {
        UserCommandType::Help => {
            bot.send_message(msg.chat.id, "A command list and introduction.").await?;
        }
        UserCommandType::Start => {
            let keyboard = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::web_app(
                "Start It!",
                WebAppInfo {
                    url: reqwest::Url::parse("https://t.me/test_tpc_bot/gamehall").expect("Failed to parse URL"),
                },
            )]]);
            bot.send_message(msg.chat.id, "Welcome to use this mini app")
                .reply_markup(keyboard)
                .await?;
        }
    }
    Ok(())
}