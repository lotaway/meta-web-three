use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

pub async fn start_future_task<ResultType, F>(task: &F) where ResultType: Send, F: TFutureTask<ResultType> {
    FutureTask::new(task, Some(Duration::from_millis(10))).await;
}

pub fn get_future_result<F: Future>(future: F) -> F::Output {
    let rt = Runtime::new().unwrap();
    rt.block_on(future)
}

#[derive(PartialEq)]
pub enum FutureTaskState {
    NoInit,
    Pending,
    Finish,
    Error,
}

pub struct FutureTaskContext<F, ResultType> {
    pub waker: Option<Waker>,
    pub state: FutureTaskState,
    pub task: &F,
    pub result: Option<ResultType>,
}

pub trait TFutureTask<ResultType>: Send + Sync {
    fn start(&mut self) -> ResultType;
}

pub struct FutureTask<ResultType: Send, F: TFutureTask<ResultType>> {
    pub duration: Option<Duration>,
    pub context: Arc<Mutex<FutureTaskContext<F, ResultType>>>,
}

impl<ResultType, F> FutureTask<ResultType, F> where ResultType: Send, F: TFutureTask<ResultType> {
    pub fn new(task: &F, duration: Option<Duration>) -> Self {
        println!("Task init");
        let context = Arc::new(Mutex::new(FutureTaskContext {
            state: FutureTaskState::NoInit,
            waker: None,
            task,
            result: None,
        }));
        let context_clone = context.clone();
        // let context_clone = Arc::clone(&context);
        let thread_handler = thread::spawn(move || {
            if let Some(d) = duration {
                thread::sleep(d);
            }
            let mut guard = context_clone.lock().unwrap();
            guard.state = FutureTaskState::Pending;
            guard.result = Some(guard.task.start());
            guard.state = FutureTaskState::Finish;
            if let Some(waker) = guard.waker.take() {
                waker.wake();
            }
        });
        match thread_handler.join() {
            Ok(result) => {
                // let res = result.unwrap();
                // dbg!("{:?}", res);
            },
            Err(error) => {

            }
        }
        Self {
            duration,
            context,
        }
    }
}

impl<ResultType, F> Future for FutureTask<ResultType, F> where ResultType: Send, F: TFutureTask<ResultType> {
    type Output = ResultType;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        println!("Start poll");
        let mut guard = self.context.lock().unwrap();
        match guard.state {
            FutureTaskState::NoInit => {
                guard.waker = Some(cx.waker().clone());
            }
            FutureTaskState::Finish => {
                println!("Task finish");
                return Poll::Ready(guard.result.unwrap());
            }
            _ => (),
        }
        if let Some(waker) = guard.waker.take() {
            if !waker.will_wake(&cx.waker()) {
                guard.waker = Some(cx.waker().clone());
            }
        }
        println!("End poll");
        Poll::Pending
    }
}