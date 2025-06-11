from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

text = "Electric cars are becoming more popular around the world. Unlike traditional cars that use gasoline or diesel, electric cars run on electricity stored in batteries. This means they do not burn fuel and do not produce harmful smoke. One big advantage of electric cars is that they are better for the environment. Since they do not release exhaust gases, they help reduce air pollution and fight climate change. Also, electric cars are quieter than regular cars, making cities less noisy. Charging an electric car is easy - you can plug it into a special charging station or even a home outlet. Depending on the car and charger, a full charge can take from 30 minutes to several hours. The range of electric cars – how far they can drive on one charge – is improving every year. Electric cars also cost less to maintain. They have fewer moving parts, so there are fewer repairs and less oil to change. However, the price of buying an electric car can be higher than a traditional one, but many countries offer financial help to make them more affordable. In the future, electric cars will play a big role in how we travel. They are a clean, quiet, and modern way to move around. If you want to help the planet and save money on fuel, an electric car might be a good choice for you."

result = evaluator.evaluate_text(text, "adults with basic or secondary education")

print(result)
