import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

const OPENAI_API_KEY = "YOUR_API_KEY"; // তোমার GPT key বসাও

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Coin Advisor (বাংলা)',
      theme: ThemeData(primarySwatch: Colors.green),
      home: CoinAdvisorPage(),
    );
  }
}

class CoinAdvisorPage extends StatefulWidget {
  @override
  _CoinAdvisorPageState createState() => _CoinAdvisorPageState();
}

class _CoinAdvisorPageState extends State<CoinAdvisorPage> {
  final TextEditingController _controller = TextEditingController();
  String _result = "";
  bool _loading = false;

  Future<void> fetchAndAnalyze(String coinSymbol) async {
    setState(() {
      _loading = true;
      _result = "";
    });

    final coin = coinSymbol.toLowerCase();
    final url =
        'https://api.coingecko.com/api/v3/coins/$coin?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false';

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final name = data['name'];
        final price = data['market_data']['current_price']['usd'];
        final change1h = data['market_data']['price_change_percentage_1h_in_currency']['usd'];
        final change24h = data['market_data']['price_change_percentage_24h_in_currency']['usd'];
        final marketCap = data['market_data']['market_cap']['usd'];

        final prompt = """
$coinSymbol কয়েন সম্পর্কিত নিচের তথ্য বিশ্লেষণ করো এবং বাংলায় বলো — এখন কিনবো, বিক্রি করবো, না হোল্ড করবো।

নাম: $name  
বর্তমান দাম: \$${price.toStringAsFixed(3)}  
১ ঘণ্টায় পরিবর্তন: ${change1h.toStringAsFixed(2)}%  
২৪ ঘণ্টায় পরিবর্তন: ${change24h.toStringAsFixed(2)}%  
মার্কেট ক্যাপ: \$${(marketCap / 1e9).toStringAsFixed(2)}B

ট্রেডারের মতো আত্মবিশ্বাসী পরামর্শ দাও, ভয় না দেখিয়ে। সংক্ষেপে বলো।
""";

        final aiResponse = await http.post(
          Uri.parse("https://api.openai.com/v1/chat/completions"),
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $OPENAI_API_KEY',
          },
          body: jsonEncode({
            "model": "gpt-3.5-turbo",
            "messages": [
              {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
          }),
        );

        if (aiResponse.statusCode == 200) {
          final aiData = jsonDecode(aiResponse.body);
          setState(() {
            _result = aiData['choices'][0]['message']['content'];
          });
        } else {
          setState(() {
            _result = "AI বিশ্লেষণ আনতে সমস্যা হয়েছে (${aiResponse.statusCode})";
          });
        }
      } else {
        setState(() {
          _result = "এই কয়েনটি পাওয়া যায়নি বা CoinGecko API error (${response.statusCode})";
        });
      }
    } catch (e) {
      setState(() {
        _result = "এক্সেপশন হয়েছে: $e";
      });
    }

    setState(() {
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('AI Coin Advisor (বাংলা)')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: 'কয়েন লিখো (যেমন: sol, btc, pepe)',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 12),
            ElevatedButton(
              onPressed: _loading
                  ? null
                  : () {
                      if (_controller.text.isNotEmpty) {
                        fetchAndAnalyze(_controller.text.trim());
                      }
                    },
              child: _loading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text("🧠 AI বিশ্লেষণ দেখাও"),
            ),
            SizedBox(height: 20),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _result,
                  style: TextStyle(fontSize: 16),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
