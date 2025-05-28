const { exec } = require('child_process')

exec(
  `setx OPENAI_API_KEY ${process.env.OPENAI_API_KEY}`,
  (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing command: ${error.message}`);
      return;
    }

    // 输出命令执行结果
    console.log('Command output:', stdout)

    if (stderr) {
      console.error('Command errors:', stderr)
    }
  },
)