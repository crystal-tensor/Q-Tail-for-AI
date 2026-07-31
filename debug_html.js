const fs = require('fs');
const http = require('http');

const content = fs.readFileSync('index.html', 'utf-8');
const server = http.createServer((req, res) => {
  res.writeHead(200, {'Content-Type': 'text/html'});
  res.end(content);
});
server.listen(6225, () => {
  console.log("Server running at http://localhost:6225/");
});
