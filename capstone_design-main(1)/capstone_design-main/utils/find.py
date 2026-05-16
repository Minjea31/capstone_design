import argparse
import json

import requests
from requests import HTTPError

API_KEY = "f742615880e442b4a0e549771bbef99d"
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

# 대한민국 영역을 넉넉하게 포함하는 기본 범위입니다.
DEFAULT_MIN_X = 124.5
DEFAULT_MAX_X = 132.0
DEFAULT_MIN_Y = 33.0
DEFAULT_MAX_Y = 39.5

DISPLAY_FIELDS = [
    "coordtype",
    "datacount",
    "roadsectionid",
    "filecreatetime",
    "cctvtype",
    "cctvurl",
    "cctvresolution",
    "coordx",
    "coordy",
    "cctvformat",
    "cctvname",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ITS 국가교통정보센터 CCTV 정보를 조회합니다."
    )
    parser.add_argument("--api-key", default=API_KEY, help="ITS OpenAPI 인증키")
    parser.add_argument("--min-x", type=float, default=DEFAULT_MIN_X, help="최소 경도")
    parser.add_argument("--max-x", type=float, default=DEFAULT_MAX_X, help="최대 경도")
    parser.add_argument("--min-y", type=float, default=DEFAULT_MIN_Y, help="최소 위도")
    parser.add_argument("--max-y", type=float, default=DEFAULT_MAX_Y, help="최대 위도")
    parser.add_argument(
        "--type",
        default="all",
        help="조회 타입(all/its/ex). 기본값은 all",
    )
    parser.add_argument(
        "--cctv-type",
        default="all",
        help="CCTV 유형(1~5 또는 all). 기본값은 all",
    )
    parser.add_argument(
        "--name",
        default="",
        help="설치 장소명에 포함될 문자열 필터",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="출력할 최대 CCTV 개수",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="결과를 원본 JSON 형태로 출력",
    )
    return parser.parse_args()


def fetch_cctv_info(api_key, min_x, max_x, min_y, max_y, query_type, cctv_type):
    query_types = ["its", "ex"] if query_type == "all" else [query_type]
    cctv_types = ["1", "2", "3", "4", "5"] if str(cctv_type) == "all" else [str(cctv_type)]

    normalized = []
    seen = set()
    last_http_error = None

    for current_type in query_types:
        for current_cctv_type in cctv_types:
            params = {
                "apiKey": api_key,
                "type": current_type,
                "cctvType": current_cctv_type,
                "minX": min_x,
                "maxX": max_x,
                "minY": min_y,
                "maxY": max_y,
                "getType": "json",
            }

            try:
                response = requests.get(API_URL, params=params, timeout=15)
                response.raise_for_status()
            except HTTPError as exc:
                last_http_error = exc
                continue

            payload = response.json()
            response_body = payload.get("response", {})
            data = response_body.get("data", [])

            shared_fields = {
                key: response_body.get(key)
                for key in ("coordtype", "datacount", "roadsectionid", "filecreatetime")
                if response_body.get(key) is not None
            }

            for item in data:
                merged = dict(shared_fields)
                merged.update(item)

                unique_key = (
                    merged.get("cctvname"),
                    merged.get("coordx"),
                    merged.get("coordy"),
                    merged.get("cctvurl"),
                )
                if unique_key in seen:
                    continue

                seen.add(unique_key)
                normalized.append(merged)

    if normalized:
        return normalized

    if last_http_error is not None:
        raise RuntimeError(
            "CCTV API 호출이 모두 실패했습니다. 인증키 권한 또는 요청 파라미터(type/cctvType)를 확인하세요."
        ) from last_http_error

    raise RuntimeError("조회된 CCTV 정보가 없습니다.")


def filter_by_name(cctv_list, keyword):
    if not keyword:
        return cctv_list

    keyword = keyword.lower()
    return [
        item
        for item in cctv_list
        if keyword in str(item.get("cctvname", "")).lower()
    ]


def print_pretty(cctv_list, limit):
    shown = cctv_list[:limit]
    print(f"조회 결과: {len(cctv_list)}건")

    for index, item in enumerate(shown, start=1):
        print(f"\n[{index}] {item.get('cctvname', '이름 없음')}")
        for field in DISPLAY_FIELDS:
            print(f"{field}: {item.get(field, '')}")


def main():
    args = parse_args()
    cctv_list = fetch_cctv_info(
        api_key=args.api_key,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        query_type=args.type,
        cctv_type=args.cctv_type,
    )
    filtered = filter_by_name(cctv_list, args.name)

    if not filtered:
        raise RuntimeError("조건에 맞는 CCTV가 없습니다.")

    if args.json:
        print(json.dumps(filtered[: args.limit], ensure_ascii=False, indent=2))
        return

    print_pretty(filtered, args.limit)


if __name__ == "__main__":
    main()
