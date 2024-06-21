use std::env;
use dotenv::dotenv;
use teloxide::dispatching::UpdateHandler;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo};
use teloxide::utils::command::{BotCommands};
use teloxide::utils::commands_repl::CommandsRepl

#[derive(BotCommands, PartialEq, Debug)]
#[command(rename_rule = "lowercase", description = "These commands are supported:")]
enum UserCommandType {
    Help,
    Start,
}

pub async fn run() {
    // teloxide::handler!();
    log::info!("Starting telegram bot...");

    dotenv().ok();
    // need env TELOXIDE_TOKEN
    let bot = Bot::from_env();
    Dispatcher::builder(bot, |err: Message| {
        println!("{}", err);
    }).build().dispatch().await;
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
                    url: reqwest::Url::parse("https://t.me/test_tpc_bot/gamehall").unwrap(),
                },
            )]]);
            bot.send_message(msg.chat.id, "Welcome to use this mini app")
                .reply_markup(keyboard)
                .await?;
        }
    }
    Ok(())
}