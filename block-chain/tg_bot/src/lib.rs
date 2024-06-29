use dotenv::dotenv;
use dptree::prelude::*;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::types::{
    InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, ReplyMarkup, WebAppInfo,
};
use teloxide::types::{InputFile, Update};
use teloxide::utils::command::BotCommands;

#[derive(BotCommands, PartialEq, Debug)]
#[command(
    rename_rule = "lowercase",
    description = "These commands are supported:"
)]
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
    let bot_name =
        std::sync::Arc::<String>::from(std::env::var("BOT_NAME").expect("Can't found BOT_NAME"));
    // need env TELOXIDE_TOKEN
    // let bot = Bot::from_env();
    let tg_bot_token = std::env::var("TG_BOT_TOKEN");
    let bot = Bot::new(tg_bot_token.expect("Can't found TG_BOT_TOKEN"));
    Dispatcher::builder(
        bot,
        dptree::entry().branch(Update::filter_message().branch(
            dptree::filter(|msg: Message| msg.text().is_some()).endpoint(
                move |msg: Message, bot: Bot| {
                    let _bot_name = bot_name.clone();
                    async move {
                        if let Some(text) = msg.text() {
                            if text.starts_with('/') {
                                match UserCommandType::parse(text, _bot_name.as_ref()) {
                                    Ok(user_command) => {
                                        answer(bot, msg, user_command).await?;
                                    }
                                    Err(err) => {
                                        println!("Failed to parse command: {}", err);
                                    }
                                }
                            }
                        }
                        Result::<(), teloxide::RequestError>::Ok(())
                    }
                },
            ),
        )),
    )
    .build()
    .dispatch()
    .await;
}

async fn answer(bot: Bot, msg: Message, cmd: UserCommandType) -> ResponseResult<()> {
    match cmd {
        UserCommandType::Help => {
            bot.send_message(msg.chat.id, "A command list and introduction.")
                .await?;
        }
        UserCommandType::Start => {
            let keyboard = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::web_app(
                "Start It!",
                WebAppInfo {
                    url: reqwest::Url::parse("https://t.me/test_tpc_bot/gamehall")
                        .expect("Failed to parse URL"),
                },
            )]]);
            bot.send_photo(
                msg.chat.id,
                InputFile::url(
                    reqwest::Url::parse("https://i.imgur.com/5y5y5y5.jpg")
                        .expect("Failed to parse URL"),
                ),
            )
            .caption("Welcome to the game hall!")
            .parse_mode(ParseMode::MarkdownV2)
            .reply_markup(ReplyMarkup::InlineKeyboard(keyboard))
            .await?;
        }
    }
    Ok(())
}
