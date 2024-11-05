import asyncio
import aiohttp
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

async def send_request(session, url, body):
    start_time = time.time()
    async with session.post(url, json=body) as response:
        await response.text()
        elapsed_time = time.time() - start_time
        logging.info(f"Response Time: {elapsed_time:.4f} seconds")
        return response.status

async def performance_test(api_url, requests_per_second, duration, body):
    total_requests = requests_per_second * duration
    logging.info(f"Sending {total_requests} requests at {requests_per_second} requests per second for {duration} seconds.")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(total_requests):
            tasks.append(send_request(session, api_url, body))
            await asyncio.sleep(1 / requests_per_second)

        responses = await asyncio.gather(*tasks)
        logging.info(f"Completed {len(responses)} requests.")

if __name__ == "__main__":
    api_url = "http://192.168.0.149:8000/predict-gender"

    body = {
        "name": "Machinelearingengineer"
    }

    requests_per_second = 10
    duration = 5

    asyncio.run(performance_test(api_url, requests_per_second, duration, body))
