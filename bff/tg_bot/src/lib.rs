use dotenv::dotenv;
use dptree::prelude::*;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::{prelude::*, ApiError, RequestError};
use teloxide::types::{
    BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, ReplyMarkup, WebAppInfo,
};
use teloxide::types::{InputFile, Update};
use teloxide::utils::command::BotCommands;

// static COMMAND_START: BotCommand = BotCommand::new("start", "Start the app");

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

#[derive(Clone)]
pub struct TGBotProgram {
    bot: std::sync::Arc<tokio::sync::Mutex<Bot>>,
    template_directory: String,
    bot_name: Option<String>,
}

impl TGBotProgram {
    pub fn new() -> Self {
        dotenv().ok();
        Self::from_token(
            std::env::var("TG_BOT_TOKEN")
                .expect("Can't found TG_BOT_TOKEN")
                .as_str(),
        )
    }

    pub fn from_token(token: &str) -> Self {
        Self::from_token_and_template(token, "templates")
    }

    pub fn from_token_and_template(token: &str, template_directory: &str) -> Self {
        let bot = std::sync::Arc::new(tokio::sync::Mutex::new(Bot::new(token)));
        Self {
            bot,
            template_directory: template_directory.into(),
            bot_name: None,
        }
    }

    pub async fn run(&mut self) {
        // teloxide::handler!(println!("{:?}", update));
        println!("Starting telegram bot...");
        // self.bot.lock().unwrap().set_my_commands(vec![COMMAND_START.clone()]);
        if (self.bot_name.is_none()) {
            let _bot_name = self
                .bot
                .lock()
                .await
                .get_me()
                .await
                .unwrap()
                .username
                .clone()
                .unwrap();
            println!("Bot name: {}", _bot_name);
            self.bot_name = Some(_bot_name);
        }
        let self_arc = std::sync::Arc::new(tokio::sync::Mutex::new(self.clone()));
        self.bot.lock().await.get_updates()
        teloxide::dispatching::Dispatcher::builder(
            self.bot.lock().await.clone(),
            dptree::entry().branch(
                Update::filter_message().branch(
                    dptree::filter(|msg: Message| {
                        msg.text().is_some() && msg.text().unwrap().starts_with('/')
                    })
                    .endpoint(move |msg: Message, bot: Bot| {
                        println!("Received command: {}", msg.text().unwrap());
                        let self_arc = self_arc.clone();
                        async move {
                            let mut self_guard = self_arc.lock().await;
                            // let mut _self = self_guard.bot.lock().unwrap();
                            let bot_name = self_guard.bot_name.clone().unwrap();
                            if let Some(text) = msg.text() {
                                match UserCommandType::parse(text, bot_name.as_ref()) {
                                    Ok(user_command) => {
                                        self_guard.answer(msg, user_command, bot).await?;
                                    }
                                    Err(err) => {
                                        println!("Failed to parse command: {}", err);
                                    }
                                }
                            }
                            Result::<(), teloxide::RequestError>::Ok(())
                        }
                    }),
                ),
            ),
        )
        .build()
        .dispatch()
        .await;
        println!("Telegram bot stopped.");
    }

    async fn answer(&mut self, msg: Message, cmd: UserCommandType, mut bot: Bot) -> ResponseResult<()> {
        println!(
            "Start to answer command: {:?}, from chat id: {}",
            cmd, msg.chat.id
        );
        let send_result = match cmd {
            UserCommandType::Help => {
                bot.send_message(msg.chat.id, "A command list and introduction.")
                    .await
            }
            UserCommandType::Start => {
                let keyboard =
                    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::web_app(
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
                    .caption("Welcome to the game hall\\!")
                    .parse_mode(ParseMode::MarkdownV2)
                    .reply_markup(ReplyMarkup::InlineKeyboard(keyboard))
                    .await
            }
            _ => {
                Result::Err(RequestError::Api(ApiError::Unknown("Unknown command".into())))
            }
        };
        if let Err(error) = send_result {
            println!("Failed to answer command: {:?}, error: {}", cmd, error);
        }
        println!(
            "Finish answering command: {:?}, from chat id: {}",
            cmd, msg.chat.id
        );
        Ok(())
    }
}
