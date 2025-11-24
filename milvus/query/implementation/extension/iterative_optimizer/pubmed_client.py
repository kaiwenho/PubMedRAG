import httpx
import asyncio
from typing import Optional

async def get_publication_info(pubids: list[str], request_id: str, timeout: float=30.0, max_retries: int=3) -> dict:
    """
    Fetch publication information from the docmetadata.transltr.io API.

    Args:
        pubids: List of PUBMED and PMCIDs (e.g., ['PMID:36008391', 'PMID:8959199', 'PMC8959199']). Other ids will not work
        request_id: Unique identifier for the request (can be a placeholder for the foreseeable)
        timeout: Request timeout in seconds (default: 30.0 - increased from 4.0)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        { _meta: { n_results: N, processing time, etc. }
          results { "PMID:999": { "abstract": ..., "article_title": ..., etc.}
                    ... }
          not_found: ["blah", ...]
        }

    Raises:
        httpx.TimeoutException: If the request times out after all retries
        httpx.RequestError: If there's an error with the request
        httpx.HTTPStatusError: If the response has an HTTP error status
    """

    # If an ID starts with "PMC:" (including the colon), drop the colon so the
    # API sees e.g. "PMC12345" instead of "PMC:12345".
# In pubmed_client.py, replace the sanitized_pubids line with:
    sanitized_pubids = []
    for pid in pubids:
        # Convert to string if it's an int
        if isinstance(pid, int):
            pid = f"PMID:{pid}"
        elif not isinstance(pid, str):
            pid = str(pid)

        # Now handle the PMC: -> PMC conversion
        if pid.upper().startswith("PMC:"):
            pid = pid.replace("PMC:", "PMC", 1)

        sanitized_pubids.append(pid)

    pubids_param = ','.join(sanitized_pubids)
    url = "https://docmetadata.transltr.io/publications"
    params = {
        'pubids': pubids_param,
        'request_id': request_id
    }

    # Retry loop
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()  # Raise an exception for HTTP error status codes
                return response.json()
        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"    Timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise httpx.TimeoutException(f"Request timed out after {max_retries} attempts (timeout={timeout}s)")
        except httpx.HTTPStatusError as e:
            # Rate limiting (429) or server errors (5xx) - retry with backoff
            if e.response.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Longer backoff for rate limits: 5s, 10s, 15s
                    print(f"    HTTP {e.response.status_code} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            else:
                # Client errors (4xx) - don't retry
                raise
        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"    Request error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise httpx.RequestError(f"Request error after {max_retries} attempts: {e}")


async def is_api_healthy(timeout: float = 2.0) -> bool:
    """
    Returns True if the TranslTR DocMetadata API is responding normally.
    Returns False if it is down, slow, or unresponsive.
    """
    API_HEALTHCHECK_PMID = "PMID:36008391"
    API_URL = "https://docmetadata.transltr.io/publications"
    params = {
        "pubids": API_HEALTHCHECK_PMID,
        "request_id": "healthcheck"
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            r = await client.get(API_URL, params=params)
            if r.status_code == 200:
                return True
            else:
                # 5xx means server unhealthy
                return False
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPError):
        return False


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(get_publication_info(('PMID:36008391','PMID:36008392','PMC8959199','not_an_id'), 'bob')))
