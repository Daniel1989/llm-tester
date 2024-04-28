import { describe, it, expect } from '@jest/globals';
import { HomeController } from './home.controller';
import { AppService } from '../service/app';
import { AppSearchDTO } from '../data-transfer-object/app';

describe('HomeController', () => {
  let controller: HomeController;
  let appService: AppService;

  beforeEach(() => {
    appService = new AppService();
    controller = new HomeController();
    controller.appService = appService;
  });

  describe('list', () => {
    it('should return a list of apps', async () => {
      const mockArgs: AppSearchDTO = {
        page: 1,
        pageSize: 10,
      };

      jest.spyOn(appService, 'list').mockResolvedValue([]);

      const result = await controller.list(mockArgs);
      expect(appService.list).toHaveBeenCalledWith(mockArgs);
      expect(result).toEqual([]);
    });

    it('should handle errors', async () => {
      const mockArgs: AppSearchDTO = {
        page: 1,
        pageSize: 10,
      };

      jest.spyOn(appService, 'list').mockRejectedValue(new Error('Test error'));

      await expect(controller.list(mockArgs)).rejects.toThrow('Test error');
    });
  });
});