use dotenv::dotenv;
use dptree::prelude::*;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::types::{
    BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, ReplyMarkup, WebAppInfo,
};
use teloxide::types::{InputFile, Update};
use teloxide::utils::command::BotCommands;

static COMMAND_START: BotCommand = BotCommand::new("start", "Start the app");

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

struct TGBotProgram {
    bot: std::sync::Arc<Bot>,
    template_directory: String,
    bot_name: Option<String>,
}

impl TGBotProgram {
    fn new() -> Self {
        dotenv().ok();
        Self::from_token(
            std::env::var("TG_BOT_TOKEN")
                .expect("Can't found TG_BOT_TOKEN")
                .as_str(),
        )
    }

    fn from_token(token: &str) -> Self {
        Self::from_token_and_template(token, "templates")
    }

    fn from_token_and_template(token: &str, template_directory: &str) -> Self {
        let bot = std::sync::Arc::new(Bot::new(token));
        Self {
            bot,
            template_directory: template_directory.into(),
            bot_name: None,
        }
    }

    pub async fn run(&mut self) {
        // teloxide::handler!(println!("{:?}", update));
        log::info!("Starting telegram bot...");
        let mut self_arc = std::sync::Arc::new(std::sync::Mutex::new(self));
        self.bot.set_my_commands(vec![COMMAND_START.clone()]);
        if (self.bot_name.is_none()) {
            let _bot_name = self_arc.lock().unwrap().bot.get_me().await.unwrap().username.unwrap();
            self.bot_name = Some(_bot_name);
        }
        let self_arc = self_arc.clone();
        Dispatcher::builder(
            self.bot.clone(),
            dptree::entry().branch(Update::filter_message().branch(
                dptree::filter(|msg: Message| msg.text().is_some()).endpoint(
                        move |msg: Message, bot: Bot| {
                            let self_arc = self_arc.clone();
                            async move {
                                let mut _self = self_arc.lock().unwrap();
                                let bot_name = _self.bot_name.unwrap();
                                if let Some(text) = msg.text() {
                                    if text.starts_with('/') {
                                        match UserCommandType::parse(text, bot_name.as_ref()) {
                                            Ok(user_command) => {
                                                _self.answer( msg, user_command).await?;
                                            }
                                            Err(err) => {
                                                println!("Failed to parse command: {}", err);
                                            }
                                        }
                                    }
                                }
                                Result::<(), teloxide::RequestError>::Ok(())
                            }
                        }
                ),
            )),
        )
        .build()
        .dispatch()
        .await;
    }

    async fn answer(&mut self, msg: Message, cmd: UserCommandType) -> ResponseResult<()> {
        match cmd {
            UserCommandType::Help => {
                self.bot.send_message(msg.chat.id, "A command list and introduction.")
                    .await?;
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
                self.bot.send_photo(
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
}
