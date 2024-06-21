use dotenv::dotenv;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};
use teloxide::utils::command::BotCommands;

#[derive(BotCommands)]
#[command(rename = "lowercase", description = "These commands are supported:")]
enum Command {
    #[command(description = "Show Help")]
    Help,
    #[command(description = "Start the main function")]
    Start,
}

pub async fn run() {
    teloxide::enable_logging!();
    log::info!("Starting telegram bot...");

    dotenv().ok();
    let bot = Bot::from_env().auto_send();

    teloxide::commands_repl(bot, answer, Command::ty()).await;
}

async fn answer(bot: AutoSend<Bot>, msg: Message, cmd: Command) -> ResponseResult<()> {
    match cmd {
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions()).await?;
        }
        Command::Start => {
            let keyboard = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url(
                "Start It!",
                reqwest::Url("https://baidu.com"),
            )]]);
            bot.send_message(msg.chat.id, "Welcome to use this mini app")
                .reply_markup(keyboard)
                .await?;
        }
    }
    Ok(())
}


struct TGBotRunner {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    async fn it_works() {
        main().await;
        assert_eq!(result, 4);
    }
}
