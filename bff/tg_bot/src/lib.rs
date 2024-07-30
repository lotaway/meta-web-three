use dotenv::dotenv;
use dptree::prelude::*;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::types::{
    BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, ReplyMarkup, WebAppInfo,
};
use teloxide::types::{InputFile, Update};
use teloxide::utils::command::BotCommands;
use teloxide::{prelude::*, ApiError, RequestError};

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
    bot: Bot,
    template_directory: String,
    bot_name: Option<String>,
}

impl TGBotProgram {
    pub fn new() -> Self {
        // dotenv().ok();
        let tg_bot_token = std::env::var("TG_BOT_TOKEN").expect("Can't found TG_BOT_TOKEN");
        Self::from_token(tg_bot_token.as_str())
    }

    pub fn from_token(token: &str) -> Self {
        Self::from_token_and_template(token, "templates")
    }

    pub fn from_token_and_template(token: &str, template_directory: &str) -> Self {
        let bot = Bot::new(token);
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
        if self.bot_name.is_none() {
            let _bot_name = self.bot.get_me().await;
            match _bot_name {
                Ok(me) => {
                    let _bot_name =me.username.clone().expect("Failed to get bot username");
                    println!("Bot name: {}", &_bot_name);
                    self.bot_name = Some(_bot_name);
                    let self_arc_for_message = std::sync::Arc::new(tokio::sync::Mutex::new(self.clone()));
                    let self_arc_for_query = std::sync::Arc::new(tokio::sync::Mutex::new(self.clone()));
                    let message_listener = tokio::spawn(async move {
                        let mut self_guard = self_arc_for_message.lock().await;
                        self_guard.listen_message(self_arc_for_message.clone()).await;
                    });
                    let query_listener = tokio::spawn(async move {
                        let mut self_guard = self_arc_for_query.lock().await;
                        self_guard.listen_query(self_arc_for_query.clone()).await;
                    });
                    let (result_message, result_query) = tokio::join![
                        message_listener,
                        query_listener,
                    ];
                    match result_message {
                        Ok(_) => {
                            println!("Telegram bot started.");
                        }
                        Err(error) => {
                            println!("Failed to start telegram bot: {}", error);
                        }
                    }
                    match result_query {
                        Ok(_) => {
                            println!("Telegram bot started.");
                        }
                        Err(error) => {
                            println!("Failed to start telegram bot: {}", error);
                        }
                    }
                }
                Err(error) => {
                    println!("Failed to get bot name: {}", error);
                }
            }
        }
        println!("Telegram bot stopped.");
    }

    async fn listen_message(&mut self, self_arc: std::sync::Arc<tokio::sync::Mutex<Self>>) {
        println!("Start listening message...");
        // let self_arc = std::sync::Arc::new(tokio::sync::Mutex::new(self.clone()));
        // let self_arc = self_arc.clone();
        let bot = self.bot.clone();
        let self_arc = std::sync::Arc::clone(&self_arc);
        teloxide::dispatching::Dispatcher::builder(
            bot,
            dptree::entry().branch(
                Update::filter_message().branch(
                    dptree::filter(|msg: Message| {
                        msg.text().is_some() && msg.text().unwrap().starts_with('/')
                    })
                    .endpoint(move |msg: Message, bot: Bot| {
                        println!("Received command: {}", msg.text().unwrap());
                        let self_arc = self_arc.clone();
                        async move {
                            let self_guard = self_arc.lock().await;
                            // let mut _self = self_guard.bot.lock().unwrap();
                            let bot_name = self_guard.bot_name.clone().unwrap();
                            if let Some(text) = msg.text() {
                                match UserCommandType::parse(text, bot_name.as_ref()) {
                                    Ok(user_command) => {
                                        TGBotProgram::answer(msg, user_command, bot).await?;
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
        .await
    }

    async fn listen_query(&mut self, self_arc: std::sync::Arc<tokio::sync::Mutex<Self>>) {
        println!("Start listening query...");
        // let self_arc = std::sync::Arc::new(tokio::sync::Mutex::new(self.clone()));
        let bot = self.bot.clone();
        let self_arc = std::sync::Arc::clone(&self_arc);
        teloxide::dispatching::Dispatcher::builder(
            bot,
            dptree::entry().branch(
                Update::filter_callback_query().branch(
                    dptree::filter(|query: CallbackQuery| {
                        query.data.is_some()
                        //  && query.data.unwrap().starts_with('/')
                    })
                    .endpoint(move |query: CallbackQuery, bot: Bot| {
                        println!("Received callback query: {}", query.data.clone().unwrap());
                        let self_arc = self_arc.clone();
                        async move {
                            let self_guard = self_arc.lock().await;
                            // let mut _self = self_guard.bot.lock().unwrap();
                            let bot_name = self_guard.bot_name.clone().unwrap();
                            if let Some(data) = query.data {
                                match UserCommandType::parse(
                                    data.clone().as_ref(),
                                    bot_name.as_ref(),
                                ) {
                                    Ok(user_command) => {
                                        let query_message = query.message.clone().unwrap();
                                        bot.send_message(query_message.chat.id, data).await?;
                                        // self_guard.answer(message, user_command, bot).await?;
                                        bot.answer_callback_query(query.id).await?;
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
    }

    async fn answer(
        msg: Message,
        cmd: UserCommandType,
        bot: Bot,
    ) -> ResponseResult<()> {
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
                let keyboard = InlineKeyboardMarkup::new(vec![vec![
                    InlineKeyboardButton::web_app(
                        "Start It!",
                        WebAppInfo {
                            url: reqwest::Url::parse("https://t.me/test_tpc_bot/gamehall")
                                .expect("Failed to parse URL"),
                        },
                    ),
                    InlineKeyboardButton::callback("HELP", "/help"),
                ]]);
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
            _ => Result::Err(RequestError::Api(ApiError::Unknown(
                "Unknown command".into(),
            ))),
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
